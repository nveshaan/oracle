import pandas as pd
import numpy as np

data = pd.read_csv('data/btcusd_ta.csv').to_numpy()

np.save('data/btc_train.npy', data[3668959:6943079, 1:])
np.save('data/btc_test.npy', data[6943079:, 1:])
