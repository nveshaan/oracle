import os
import numpy as np
import pandas as pd
import torch
import ta

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class yf_Trendlines(Dataset):
    def __init__(self, test=False, fhr=30, phr=30, day=30, wek = 30, mon=30, window=10, order=3):
        super().__init__()
        if test:
            self.min = pd.read_csv('data/test/10,1 10,11 2min.csv', header=2)
            self.hr = pd.read_csv('data/test/9,10 10,11 1hr.csv', header=2)
            self.day = pd.read_csv('data/test/9,1 10,11 1day.csv', header=2)
        else:
            min1 = pd.read_csv('data/8,15 8,29 2min.csv', header=2)
            min2 = pd.read_csv('data/9,2 9,29 2min.csv', header=2)
            self.min = pd.concat((min1, min2))
            self.hr = pd.read_csv('data/8,4 9,29 1hr.csv', header=2)
            self.day = pd.read_csv('data/7,2 9,29 1day.csv', header=2)

        self.order = order
        def transform(df):
            numeric_cols = df.columns[1:6]
            for col in numeric_cols:
                df[col] = ta.trend.ema_indicator(df[col], window=window)
            return df

        self.min = transform(self.min)
        self.hr = transform(self.hr)
        self.day = transform(self.day)

        self.min = self.min.dropna().to_numpy()
        self.hr = self.hr.dropna().to_numpy()
        self.day = self.day.dropna().to_numpy()

        self.fhr_index = [[i+j for j in range(fhr)] for i in range(6*day+phr, len(self.min)-fhr)]
        self.phr_index = [[i+j for j in range(phr)] for i in range(6*day, len(self.min)-fhr-phr)]
        self.day_index = [[i+j for j in range(6*day)] for i in range(0, len(self.min)-fhr-day)]

        self.week_index = np.arange(wek)
        self.mon_index = np.arange(mon)

    def __len__(self):
        return len(self.fhr_index)

    def __getitem__(self, idx):
        fhr = self.min[self.fhr_index[idx]]
        phr = self.min[self.phr_index[idx]]
        day = self._min_to_hr(self.min[self.day_index[idx]])

        date = fhr[0, 0].split()[0]
        week_id = np.where(self.hr[:, 0] <= date)[0][-1]
        mon_id = np.where(self.day[:, 0] <= date)[0][-1]

        week = self.hr[week_id-self.week_index]
        mon = self.day[mon_id-self.mon_index]

        base_fhr = np.float32(fhr[0, 4])
        base_phr = np.float32(phr[0, 4])
        base_day = np.float32(day[0, 4])
        base_week = np.float32(week[0, 4])
        base_mon = np.float32(mon[0, 4])

        fhr_tensor  = torch.tensor(self.tanh(fhr[:, 1].astype(np.float32)  - base_fhr)).unsqueeze(0)
        phr_tensor  = torch.tensor(self.tanh(phr[:, 1].astype(np.float32)  - base_phr))
        day_tensor  = torch.tensor(self.tanh(day[:, 1].astype(np.float32) - base_day))
        week_tensor = torch.tensor(self.tanh(week[:, 1].astype(np.float32) - base_week))
        mon_tensor  = torch.tensor(self.tanh(mon[:, 1].astype(np.float32)  - base_mon))

        sin = transforms.Lambda(lambda x: torch.sin(x))
        cos = transforms.Lambda(lambda x: torch.cos(x))

        temp = [torch.ones_like(fhr_tensor)]
        for i in range(self.order):
            temp.append(sin(fhr_tensor*(i+1)))
            temp.append(cos(fhr_tensor*(i+1)))

        temp = np.array(temp)
        fhr_tensor = torch.cat([torch.tensor(temp, dtype=torch.float32)], dim=0)

        cond_rows = torch.tensor(np.array([phr_tensor, day_tensor, week_tensor, mon_tensor]))
        temp = [torch.ones_like(cond_rows)]
        for i in range(self.order):
            temp.append(sin(cond_rows*(i+1)))
            temp.append(cos(cond_rows*(i+1)))

        temp = np.array(temp)    
        condition = torch.cat([torch.tensor(temp, dtype=torch.float32)], dim=0)

        return fhr_tensor.float(), condition.float()

    def _min_to_hr(self, series):
        return np.array([
            [series[i+5, 1], max(series[i:i+6, 2]), min(series[i:i+6, 3]), series[i, 4], sum(series[i:i+6, 5])]
            for i in range(0, len(series)-5, 6)
        ])
    
    def tanh(self, x):
        return np.tanh(x*0.01)*np.pi
    

class btc_Trendlines(Dataset):
    # Data split boundaries (row indices in CSV)
    TRAIN_START = 3668959
    TRAIN_END = 6943079
    TEST_START = 6943079
    
    def __init__(self, test=False, order=3, seq_len=300, pred_len=60, cache_path='data/btcusd_ta.npy'):
        super().__init__()
        self.test = test
        self.order = order
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Convert CSV to memory-mapped numpy array (one-time cost)
        if not os.path.exists(cache_path):
            print("Converting CSV to numpy memmap (one-time)...")
            # Read in chunks to avoid memory spike
            chunks = pd.read_csv('data/btcusd_ta.csv', chunksize=100000, usecols=lambda c: c != 'Date')
            first_chunk = next(chunks)
            n_cols = first_chunk.shape[1]
            
            # Count total rows first
            total_rows = len(first_chunk)
            for chunk in chunks:
                total_rows += len(chunk)
            
            # Create memmap and fill it
            chunks = pd.read_csv('data/btcusd_ta.csv', chunksize=100000, usecols=lambda c: c != 'Date')
            mmap = np.memmap(cache_path, dtype=np.float32, mode='w+', shape=(total_rows, n_cols))
            offset = 0
            for chunk in chunks:
                # Replace inf/nan and clip to safe float32 range
                arr = chunk.to_numpy(dtype=np.float64)
                arr = np.nan_to_num(arr, nan=0.0, posinf=1e30, neginf=-1e30)
                arr = np.clip(arr, -1e30, 1e30)
                mmap[offset:offset + len(arr)] = arr.astype(np.float32)
                offset += len(arr)
            mmap.flush()
            del mmap
            print(f"Saved memmap: {total_rows} rows, {n_cols} cols")
        
        # Load as memory-mapped (doesn't load into RAM)
        # First, get shape from a quick read
        temp = np.memmap(cache_path, dtype=np.float32, mode='r')
        n_cols = 91  # TA features count (adjust if different)
        n_rows = len(temp) // n_cols
        del temp
        
        self.data = np.memmap(cache_path, dtype=np.float32, mode='r', shape=(n_rows, n_cols))
        
        # Set data range based on train/test split
        if test:
            self.start_idx = self.TEST_START
            self.end_idx = n_rows
        else:
            self.start_idx = self.TRAIN_START
            self.end_idx = self.TRAIN_END
        
        # Store only start indices (not full index lists)
        # Skip n rows between samples to reduce correlation
        self.stride = 30
        usable_rows = self.end_idx - self.start_idx - seq_len - pred_len
        self.n_samples = usable_rows // self.stride
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Compute indices on-the-fly instead of storing lists
        # Each sample starts at start_idx + idx * stride
        p_start = self.start_idx + idx * self.stride
        p_end = p_start + self.seq_len
        f_start = p_end
        f_end = f_start + self.pred_len
        
        ptrend = np.array(self.data[p_start:p_end])  # Copy from memmap
        ftrend = np.array(self.data[f_start:f_end, 40])  # Only column 40

        # Normalize each feature (column) across time (axis=0)
        min_vals = ptrend.min(axis=0, keepdims=True)
        max_vals = ptrend.max(axis=0, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        ptrend = (ptrend - min_vals) / range_vals

        min_val = ftrend.min()
        ftrend = np.tanh((ftrend - min_val) * 0.01) * np.pi

        # Convert to tensor
        ftrend_tensor = torch.from_numpy(ftrend).float().unsqueeze(0)  # (1, 60)
        ptrend_tensor = torch.from_numpy(ptrend).float()  # (300, 91)

        # Build Fourier features for ftrend
        temp = [torch.ones_like(ftrend_tensor)]
        for i in range(self.order):
            temp.append(torch.sin(ftrend_tensor * (i + 1)))
            temp.append(torch.cos(ftrend_tensor * (i + 1)))

        ftrend_out = torch.cat(temp, dim=0).unsqueeze(1)  # (7, 1, 60)

        return ftrend_out, ptrend_tensor


if __name__ == '__main__':
    ds = btc_Trendlines()
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(dl))
    print("Batch ftrend tensor shape:", batch[0].shape)
    print("Batch ptrend tensor shape:", batch[1].shape)