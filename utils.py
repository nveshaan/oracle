import torch
from torch.utils.data import DataLoader
from dataset import yf_Trendlines
from diffusion import create_diffusion
from model.network import UNet
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# ------------------ Setup ------------------
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
num_samples = 1
# seed = 1000
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.set_grad_enabled(False)

# Load model
model = UNet([181, 1024, 512, 256, 128, 64, 128, 256, 512, 1024, 362]).to(device)
ckpt_path = "checkpoints/1010_1204_model.pth"
state = torch.load(ckpt_path, map_location=device)
if isinstance(state, dict) and 'model' in state:
    state = state['model']
model.load_state_dict(state, strict=False)
model.eval()
diffusion = create_diffusion(timestep_respacing="")

# Dataset / single batch
dataset = yf_Trendlines()
loader = DataLoader(dataset, batch_size=30, shuffle=True)
x, y = list(loader)[0]
x = x.repeat(num_samples, 1)
y = y.repeat(num_samples, 1, 1, 1)
x = x.to(device, dtype=torch.float32)
y = y.to(device, dtype=torch.float32)

with torch.no_grad():
    z = model.y_embedder(y)
    print(z.norm().item(), z.max().item(), z.mean().item())
    print(z.shape)

# ema of z
z = z.unfold(1, 150, 1).mean(dim=2)

plt.plot(z.T.cpu().numpy(), alpha=0.3)
plt.show()