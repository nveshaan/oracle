import torch
from torch.utils.data import DataLoader
from dataset import yf_Trendlines
from diffusion import create_diffusion
from models import DiT_models
import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------ Setup ------------------
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
num_samples = 10
# seed = 1000
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.set_grad_enabled(False)

# Load model
model = DiT_models["DiT-L"](input_size=(1, 30)).to(device)
ckpt_path = "checkpoints/DiT_recon_fourier_model.pth"
state = torch.load(ckpt_path, map_location=device)
if isinstance(state, dict) and 'model' in state:
    state = state['model']
model.load_state_dict(state, strict=False)
model.eval()
diffusion = create_diffusion(timestep_respacing="")

# Dataset / single batch
dataset = yf_Trendlines(test=True)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
x, y = list(loader)[88]
x = x.repeat(num_samples, 1, 1, 1)
y = y.repeat(num_samples, 1, 1, 1)
x = x.to(device, dtype=torch.float32)
y = y.to(device, dtype=torch.float32)

# --------------- Full reverse diffusion sampling (prediction) ---------------
with torch.no_grad():
    sample = diffusion.p_sample_loop(
        model,  # model callable
        x.shape,  # target shape
        torch.randn_like(x),  # initial noise
        model_kwargs={"y": y},
        progress=True,
        device=device,
        clip_denoised=True,
    )

x = x.cpu().numpy()
sample = sample.cpu().numpy()

x_ = (np.arcsin(x[:, 1]) + np.arccos(x[:, 2]) + np.arcsin(x[:, 3]) + np.arccos(x[:, 4]) + np.arcsin(x[:, 5]) + np.arccos(x[:, 6]))/12
sample_ = (np.arcsin(sample[:, 1]) + np.arccos(sample[:, 2]) + np.arcsin(sample[:, 3]) + np.arccos(sample[:, 4]) + np.arcsin(sample[:, 5]) + np.arccos(sample[:, 6]))/12

gt = np.arctanh(x_[0, :].squeeze()/np.pi)/0.01
pred = np.arctanh(sample_[:, :].squeeze()/np.pi)/0.01

for i in range(num_samples):
    plt.plot(pred[i], label='Predicted', color='orange', alpha=0.3)
plt.plot(gt, label='Ground Truth')
plt.tight_layout()
plt.show()
