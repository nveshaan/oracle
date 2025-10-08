import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

class yf_Trendlines(Dataset):
    def __init__(self, dir='data', fhr=30, phr=30, day=30, wek = 30, mon=30):
        super().__init__()
        min1 = pd.read_csv('data/8,15 8,29 2min.csv', header=2)
        min2 = pd.read_csv('data/9,2 9,29 2min.csv', header=2)
        self.min = pd.concat((min1, min2)).to_numpy()
        self.hr = pd.read_csv('data/8,4 9,29 1hr.csv', header=2).to_numpy()
        self.day = pd.read_csv('data/7,2 9,29 1day.csv', header=2).to_numpy()

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

        date = fhr[0][0].split()[0]
        week_id = np.where(self.hr[:, 0] <= date)[0][-1]
        mon_id = np.where(self.day[:, 0] <= date)[0][-1]

        week = self.hr[week_id-self.week_index]
        mon = self.day[mon_id-self.mon_index]

        base_fhr = np.float32(fhr[0, 4])
        base_phr = np.float32(phr[0, 4])
        base_week = np.float32(week[0, 4])
        base_mon = np.float32(mon[0, 4])

        fhr_tensor  = torch.tensor(self.tanh(fhr[:, 1:].astype(np.float32)  - base_fhr))
        phr_tensor  = torch.tensor(self.tanh(phr[:, 1:].astype(np.float32)  - base_phr))
        day_np = day.astype(np.float32)
        base_day = np.float32(day_np[0, 4])
        day_tensor  = torch.from_numpy(self.tanh(day_np) - base_day)  # shape (N_hr, 5)
        week_tensor = torch.tensor(self.tanh(week[:,1:].astype(np.float32) - base_week))
        mon_tensor  = torch.tensor(self.tanh(mon[:,1:].astype(np.float32)  - base_mon))

        # fhr as (1, 5, T) image-like tensor (features vertically, time horizontally)
        fhr_tensor = fhr_tensor.T.unsqueeze(0)  # (1, 5, T)

        cond_rows = [phr_tensor.T, day_tensor.T, week_tensor.T, mon_tensor.T]  # each (5, T)
        condition = torch.cat(cond_rows, dim=0).unsqueeze(0)  # (1,20,T)

        return fhr_tensor.float(), condition.float()

    def _min_to_hr(self, series):
        return np.array([
            [series[i+5, 1], max(series[i:i+6, 2]), min(series[i:i+6, 3]), series[i, 4], sum(series[i:i+6, 5])]
            for i in range(0, len(series)-5, 6)
        ])
    
    def tanh(self, x):
        return np.tanh(x*0.01)

if __name__ == '__main__':
    ds = yf_Trendlines()
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(dl))
    print("Batch fhr tensor shape (C,5,T):", batch[0].shape)
    print("Batch condition tensor shape (C,20,T):", batch[1].shape)