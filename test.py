import torch
from torch.utils.data import DataLoader
from dataset import yf_Trendlines
from diffusion import create_diffusion
from models import DiT_models
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import gaussian_kde

# ------------------ Setup ------------------
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
num_samples = 100
# seed = 1000
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.set_grad_enabled(False)

# Load model
model = DiT_models["DiT-XL"](input_size=(1, 30)).to(device)
ckpt_path = "checkpoints/DiT_XL_postLN.pth"
state = torch.load(ckpt_path, map_location=device)
if isinstance(state, dict) and 'model' in state:
    state = state['model']
model.load_state_dict(state, strict=False)
model.eval()
diffusion = create_diffusion(timestep_respacing="")

# Dataset / single batch
dataset = yf_Trendlines(test=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
x, y = list(loader)[100]
x = x.repeat(num_samples, 1, 1, 1)
y = y.repeat(num_samples, 1, 1, 1)

x = x.to(device, dtype=torch.float32)
y = y.to(device, dtype=torch.float32)

x = torch.cat([x, x], 0)
y_null = model.y_embedder.null_trendline.expand(y.shape[0], -1, -1, -1).to(device)
y = torch.cat([y, y_null], 0)

model_kwargs = dict(y=y, cfg_scale=4.0)

# --------------- Full reverse diffusion sampling (prediction) ---------------
with torch.no_grad():
    sample = diffusion.ddim_sample_loop(
        model.forward_with_cfg,  # model callable
        x.shape,  # target shape
        torch.randn_like(x),  # initial noise
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
        clip_denoised=True,
    )

x, _ = x.chunk(2, dim=0)
sample, _ = sample.chunk(2, dim=0)

x = x.cpu().numpy()
sample = sample.cpu().numpy()

x_ = (np.arcsin(x[:, 1]) + np.arccos(x[:, 2]) + np.arcsin(x[:, 3]) + np.arccos(x[:, 4]) + np.arcsin(x[:, 5]) + np.arccos(x[:, 6]))/12
sample_ = (np.arcsin(sample[:, 1]) + np.arccos(sample[:, 2]) + np.arcsin(sample[:, 3]) + np.arccos(sample[:, 4]) + np.arcsin(sample[:, 5]) + np.arccos(sample[:, 6]))/12

gt = np.arctanh(x_[0, :].squeeze()/np.pi)/0.01
pred = np.arctanh(sample_[:, :].squeeze()/np.pi)/0.01

for i in range(num_samples):
    plt.plot(pred[i], label='Predicted', color='orange', alpha=0.2)
plt.plot(gt, label='Ground Truth')
plt.tight_layout()
plt.show()

# KDE heatmap for each time step
time_steps = np.arange(30)
v_min, v_max = pred.min(), pred.max()
v_grid = np.linspace(v_min, v_max, 100)  # 100 bins for values
density = np.zeros((30, 100))

for t in time_steps:
    values = pred[:, t]
    if len(np.unique(values)) > 1:  # avoid singular matrix for KDE
        kde = gaussian_kde(values)
        density[t, :] = kde(v_grid)
    else:
        # if all values are the same, place density at that value
        idx = np.argmin(np.abs(v_grid - values[0]))
        density[t, idx] = 1.0

# Plot the heatmap
fig, ax = plt.subplots(figsize=(10, 6))
c = ax.pcolormesh(time_steps, v_grid, density.T, shading='gouraud', cmap='viridis')
ax.set_xlabel('Time Step')
ax.set_ylabel('Value')
ax.set_title('KDE Heatmap of Predicted Values per Time Step')
plt.colorbar(c, ax=ax, label='Density')
plt.tight_layout()
plt.show()

