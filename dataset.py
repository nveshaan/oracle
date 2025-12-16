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
    def __init__(self, order=3):
        super().__init__()
        self.order = order
        self.data = pd.read_csv('data/btcusd_ta.csv')
        self.data = self.data.dropna().to_numpy()[3668959:, 1:]
        self.pindex = [[i+j for j in range(300)] for i in range(0, len(self.data)-360)]
        self.findex = [[i+j for j in range(60)] for i in range(300, len(self.data)-60)]

    def __len__(self):
        return len(self.pindex)

    def __getitem__(self, idx):
        ptrend = self.data[self.pindex[idx]]
        ftrend = self.data[self.findex[idx], 40]  # (60,)

        # Normalize each feature (column) across time (axis=0)
        min_vals = np.nanmin(ptrend, axis=0, keepdims=True)
        max_vals = np.nanmax(ptrend, axis=0, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        ptrend = (ptrend - min_vals) / range_vals  # (300, num_features)

        min_val = np.nanmin(ftrend)
        ftrend = np.tanh((ftrend - min_val) * 0.01) * np.pi  # (60,)

        # Convert to tensor
        ftrend_tensor = torch.tensor(ftrend, dtype=torch.float32).unsqueeze(0)  # (1, 60)
        ptrend_tensor = torch.tensor(ptrend, dtype=torch.float32)  # (300, num_features)

        # Build Fourier features for ftrend
        temp = [torch.ones(1, ftrend_tensor.shape[1])]  # (1, 60)
        for i in range(self.order):
            temp.append(torch.sin(ftrend_tensor * (i + 1)))  # (1, 60)
            temp.append(torch.cos(ftrend_tensor * (i + 1)))  # (1, 60)

        ftrend_out = torch.cat(temp, dim=0)  # (1 + 2*order, 60) = (7, 60) if order=3

        return ftrend_out, ptrend_tensor  # (7, 60), (300, num_features)


if __name__ == '__main__':
    ds = btc_Trendlines()
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(dl))
    print("Batch ftrend tensor shape (C,1,T):", batch[0].shape)
    print("Batch ptrend tensor shape (C,300,T):", batch[1].shape)