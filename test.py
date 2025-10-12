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
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.set_grad_enabled(False)

# Load model
model = DiT_models["DiT-L"](input_size=(5, 30)).to(device)
ckpt_path = "checkpoints/1009_1318_model.pth"
state = torch.load(ckpt_path, map_location=device)
if isinstance(state, dict) and 'model' in state:
    state = state['model']
model.load_state_dict(state, strict=False)
model.eval()
diffusion = create_diffusion(timestep_respacing="")  # default schedule

# Dataset / single batch
dataset = yf_Trendlines()
loader = DataLoader(dataset, batch_size=1, shuffle=False)
x, cond = list(loader)[1]  # x: (B,1,5,30)  cond: (B,1,20,30)
x = x.repeat(num_samples, 1, 1, 1)
x = x.to(device, dtype=torch.float32)
cond = cond.repeat(num_samples, 1, 1, 1)
cond = cond.to(device, dtype=torch.float32)

# --------------- Full reverse diffusion sampling (prediction) ---------------
with torch.no_grad():
    # Use the diffusion sampler to generate a full sample conditioned on cond
    sample = diffusion.p_sample_loop(
        model,  # model callable
        x.shape,  # target shape
        torch.randn_like(x),  # initial noise
        model_kwargs={"y": cond},
        progress=True,
        device=device,
        clip_denoised=True,
    )

gt = np.arctanh(x[0, 0, 0, :].squeeze().cpu().numpy())/0.01
pred = np.arctanh(sample[:, 0, 0, :].squeeze().cpu().numpy())/0.01

for i in range(num_samples):
    plt.plot(pred[i], label='Predicted', color='orange', alpha=0.3)
plt.plot(gt, label='Ground Truth')
plt.tight_layout()
plt.show()
