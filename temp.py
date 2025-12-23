import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ts2vec import TS2Vec
from sklearn.decomposition import PCA

import torch
from diffusion import create_diffusion
from models import DiT_models

device = 'mps'

# data = pd.read_csv('data/btcusd_ta.csv').to_numpy()[3668959:, 1:]
data = np.load('data/btc_train.npy')
data = data[: (len(data) // 300) * 300].reshape(-1, 300, data.shape[1])
# Per-instance min-max normalization (scales each sample to [0, 1])
min_vals = np.nanmin(data, axis=1, keepdims=True)
max_vals = np.nanmax(data, axis=1, keepdims=True)
range_vals = max_vals - min_vals
range_vals[range_vals == 0] = 1  # Avoid division by zero
data = (data - min_vals) / range_vals
data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
# model = TS2Vec(
#     input_dims=data.shape[-1],
#     output_dims=384,
#     device=device,
# )
# model.load('training/btcusd_ta.csv__dit_s_20251216_153223/model.pkl')

model = DiT_models["DiT-S"](input_size=(1, 60)).to(device)
ckpt_path = "checkpoints/btc_logprice.pth"
state = torch.load(ckpt_path, map_location=device)
if isinstance(state, dict) and 'model' in state:
    state = state['model']
model.load_state_dict(state, strict=False)
model.eval()
diffusion = create_diffusion(timestep_respacing="")

repr = model.y_embedder.ts2vec.encode(torch.from_numpy(data).to(device), encoding_window='full_series')
repr = repr.cpu().numpy()
print('Representation shape:', repr.shape)

plt.plot(repr.T, alpha=0.3)
plt.title('TS2Vec Representations for BTC-USD Time Series Data')
plt.tight_layout()
plt.show()
