import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.network import UNet
from diffusion import create_diffusion
from dataset import yf_Trendlines

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import random
import datetime


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_epoch(loader, network, diffusion, loss_fn, optimizer, timesteps, device, epoch, log_to_wandb):
    network.train()
    total_loss = 0.0

    loop = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch_idx, batch in enumerate(loop):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        t = torch.randint(0, diffusion.num_timesteps, (len(x),), device=device).long()

        model_kwargs = dict(y=y)
        loss_dict = diffusion.training_losses(network, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        if log_to_wandb:
            wandb.log({"train/loss": loss.item()}, step=epoch * len(loader) + batch_idx)

    return total_loss / len(loader)

def validate_epoch(loader, network, diffusion, loss_fn, timesteps, device, epoch):
    network.eval()
    total_loss = 0.0

    with torch.inference_mode():
        loop = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        for batch_idx, batch in enumerate(loop):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (len(x),), device=device)

            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(network, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

@hydra.main(config_path="configs", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if cfg.wandb.log:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg))

    dataset = yf_Trendlines()

    val_size = int(len(dataset) * cfg.data.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=False)

    nodes = cfg.model.nodes
    model = UNet(nodes).to(device)
    if cfg.train.use_compile:
        model = torch.compile(model)

    timesteps = cfg.model.timesteps
    diffusion = create_diffusion(timestep_respacing="")

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Dataset Size: {len(dataset)} | Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

    for epoch in range(cfg.train.epochs):
        print(f"\n Epoch {epoch+1}/{cfg.train.epochs}")
        train_loss = train_epoch(train_loader, model, diffusion, loss_fn, optimizer, timesteps, device, epoch, cfg.wandb.log)
        #val_loss = validate_epoch(val_loader, model, diffusion, loss_fn, timesteps, device, epoch)

        print(f" Train Loss: {train_loss:.6f} ")#| Val Loss: {val_loss:.6f}")

        if cfg.train.epoch_save:
            os.makedirs(os.path.dirname(f'checkpoints/{datetime.datetime.now().strftime("%m%d_%H%M")}_epoch{epoch+1}.pth'), exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/{datetime.datetime.now().strftime("%m%d_%H%M")}_epoch{epoch+1}.pth')
            print(f"\nCheckpoint saved successfully.")

        if cfg.wandb.log:
            wandb.log({
                "epoch": epoch + 1,
                "train/avg_loss": train_loss,
                #"val/avg_loss": val_loss
            })

    print("Training complete.")

    if cfg.train.save:
        os.makedirs(os.path.dirname(f'checkpoints/{datetime.datetime.now().strftime("%m%d_%H%M")}_model.pth'), exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/{datetime.datetime.now().strftime("%m%d_%H%M")}_model.pth')
        print(f"\nModel saved successfully.")

    if cfg.wandb.log:
        wandb.finish()


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
    print("\n")
    main()