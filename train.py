# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
from torch.utils.data import DataLoader
from dataset import btc_Trendlines
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import os
import random
from tqdm import tqdm
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import datetime

from models import DiT_models
from diffusion import create_diffusion


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

@hydra.main(config_path="configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Trains a new DiT model.
    """
    device = 'mps'
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.wandb.log:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg))

    # Setup an experiment folder:
    os.makedirs(cfg.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{cfg.results_dir}/*"))
    model_string_name = cfg.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{cfg.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Experiment directory created at {experiment_dir}")

    # Create model:
    trendline_size = (1, 60)
    model = DiT_models[cfg.model](
        input_size=trendline_size,
        model = cfg.model,
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = model.to(device)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0)

    dataset = btc_Trendlines(order=3, seq_len=300, pred_len=60)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    print(f"Dataset contains {len(dataset):,} trendlines ({cfg.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    print(f"Training for {cfg.epochs} epochs...")
    for epoch in range(cfg.epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch_idx, batch in enumerate(loop):
            x, cond = batch
            x = x.to(device, dtype=torch.float32)
            cond = cond.to(device, dtype=torch.float32)
            t = torch.randint(0, diffusion.num_timesteps, (x.size(0),), device=device)
            model_kwargs = dict(y=cond)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            loop.set_postfix(loss=loss.item())
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % cfg.log_every == 0:
                # Measure training speed:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()
                if cfg.wandb.log:
                    wandb.log({
                        "step": train_steps,
                        "train loss": avg_loss,
                        "train steps/sec": steps_per_sec
                    })
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % cfg.ckpt_every == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    if cfg.wandb.log:
        wandb.finish()

    os.makedirs(os.path.dirname(f'checkpoints/{datetime.datetime.now().strftime("%m%d_%H%M")}_model.pth'), exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/{datetime.datetime.now().strftime("%m%d_%H%M")}_model.pth')
    print(f"\nModel saved successfully.")

if __name__ == "__main__":
    print("PyTorch Version:", torch.__version__)
    if torch.cuda.is_available():
        print("Using CUDA")
        print("CUDA Available:", torch.cuda.is_available())
        print("CUDA Version:", torch.version.cuda)
        print("cuDNN Version:", torch.backends.cudnn.version())
        print("Device Count:", torch.cuda.device_count())
    elif torch.backends.mps.is_available():
        print("Using MPS")
        print("MPS Available:", torch.backends.mps.is_available())
        print("MPS Built:", torch.backends.mps.is_built())
        print("Device Count:", torch.mps.device_count())
    else:
        print("Using CPU")
        print("Device:", torch.cpu.current_device())
    main()
