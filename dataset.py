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

        fhr_tensor  = torch.tensor(fhr[:,1:].astype(float)-fhr[0, 4])
        phr_tensor  = torch.tensor(phr[:,1:].astype(float)-phr[0, 4])
        day_tensor  = torch.tensor(day-day[0, 4], dtype=float)
        week_tensor = torch.tensor(week[:,1:].astype(float)-week[0, 4])
        mon_tensor  = torch.tensor(mon[:,1:].astype(float)-mon[0, 4])
        return fhr_tensor, (phr_tensor, day_tensor, week_tensor, mon_tensor)

    def _min_to_hr(self, series):
        return np.array([
            [series[i+5, 1], max(series[i:i+6, 2]), min(series[i:i+6, 3]), series[i, 4], sum(series[i:i+6, 5])]
            for i in range(0, len(series)-5, 6)
        ])
        

if __name__ == '__main__':
    ds = yf_Trendlines()
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(dl))
    print("Batch fhr tensor shape:", batch[0].shape)