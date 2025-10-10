import torch
from torch.utils.data import DataLoader
from dataset import yf_Trendlines
from model.diffusion import GaussianDiffusion
from model.network import UNet
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# ------------------ Setup ------------------
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
num_samples = 100
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.set_grad_enabled(False)

# Load model
model = UNet([181, 1024, 512, 256, 128, 64, 128, 256, 512, 1024, 181]).to(device)
ckpt_path = "checkpoints/1010_1038_model.pth"
state = torch.load(ckpt_path, map_location=device)
if isinstance(state, dict) and 'model' in state:
    state = state['model']
model.load_state_dict(state, strict=False)
model.eval()
diffusion = GaussianDiffusion(1000)

# Dataset / single batch
dataset = yf_Trendlines()
loader = DataLoader(dataset, batch_size=1, shuffle=False)
x, y = list(loader)[1000]
x = x.repeat(num_samples, 1)
y = y.repeat(num_samples, 1, 1, 1)
x = x.to(device, dtype=torch.float32)
y = y.to(device, dtype=torch.float32)

# --------------- Full reverse diffusion sampling (prediction) ---------------
with torch.no_grad():
    x_t = x.clone()
    loop = tqdm(range(diffusion.T-1, -1, -1), total=diffusion.T, desc='Sampling')
    for i in loop:
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        x_t = diffusion.sample_timestep(model, x_t, t, y)
        if i % 100 == 0 or i == diffusion.T - 1:
            loop.set_postfix(timestep=i)
    sample = x_t

x = x.cpu().numpy()
sample = sample.cpu().numpy()

x_ = (np.arcsin(x[:, 0:30]) + np.arccos(x[:, 30:60]) + np.arcsin(x[:, 60:90]) + np.arccos(x[:, 90:120]) + np.arcsin(x[:, 120:150]) + np.arccos(x[:, 150:180]))/12
sample_ = (np.arcsin(sample[:, 0:30]) + np.arccos(sample[:, 30:60]) + np.arcsin(sample[:, 60:90]) + np.arccos(sample[:, 90:120]) + np.arcsin(sample[:, 120:150]) + np.arccos(sample[:, 150:180]))/12

gt = np.arctanh(x_[0, :].squeeze()/np.pi)/0.01
pred = np.arctanh(sample_[:, :].squeeze()/np.pi)/0.01

for i in range(num_samples):
    plt.plot(pred[i], label='Predicted', color='orange', alpha=0.3)
plt.plot(gt, label='Ground Truth')
plt.tight_layout()
plt.show()
