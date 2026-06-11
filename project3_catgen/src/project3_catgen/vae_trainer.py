import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from project3_catgen.generation import generate_images, sample_latent_vae


class VAETrainer:
    def __init__(
        self,
        vae: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str | Path,
        latent_dim: int,
        sample_grid_size: int = 64,
        beta: float = 1.0,
    ) -> None:
        if sample_grid_size < 1:
            raise ValueError("sample_grid_size must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")

        self.vae = vae
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.latent_dim = latent_dim
        self.beta = beta
        self.fixed_noise = sample_latent_vae(sample_grid_size, latent_dim, device)
        self.start_epoch = 0

    def train_one_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.vae.train()

        totals = {"reconstruction_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0}
        total_samples = 0

        progress = tqdm(dataloader, desc="train", leave=False)
        for batch in progress:
            real_images = batch[0] if isinstance(batch, (list, tuple)) else batch
            real_images = real_images.to(self.device, non_blocking=True)
            batch_size = real_images.size(0)

            self.optimizer.zero_grad(set_to_none=True)
            recon, mu, log_var = self.vae(real_images)

            recon_loss = F.mse_loss(recon, real_images, reduction="mean")
            kl_loss = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())
            total_loss = recon_loss + self.beta * kl_loss

            total_loss.backward()
            self.optimizer.step()

            totals["reconstruction_loss"] += recon_loss.item() * batch_size
            totals["kl_loss"] += kl_loss.item() * batch_size
            totals["total_loss"] += total_loss.item() * batch_size
            total_samples += batch_size

            progress.set_postfix(
                recon=f"{recon_loss.item():.3f}",
                kl=f"{kl_loss.item():.4f}",
            )

        if total_samples == 0:
            raise ValueError("Training dataloader is empty")
        return {name: value / total_samples for name, value in totals.items()}

    def save_samples(self, epoch: int) -> Path:
        images = generate_images(
            self.vae.decode,
            self.fixed_noise,
            batch_size=self.fixed_noise.size(0),
            module=self.vae,
        )
        sample_dir = self.checkpoint_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_path = sample_dir / f"epoch_{epoch + 1:04d}.png"
        save_image(
            images,
            output_path,
            nrow=math.ceil(math.sqrt(images.size(0))),
            normalize=True,
            value_range=(-1, 1),
        )
        return output_path

    def save_checkpoint(self, filename: str, epoch: int) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.checkpoint_dir / f"{filename}.pth"
        temp_path = output_path.with_name(f"{output_path.name}.tmp")
        torch.save(
            {
                "epoch": epoch,
                "vae_state_dict": self.vae.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "fixed_noise": self.fixed_noise.cpu(),
                "rng_state": {
                    "torch": torch.get_rng_state(),
                    "cuda": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            },
            temp_path,
        )
        temp_path.replace(output_path)
        return output_path

    def load_checkpoint(self, filename: str, restore_rng: bool = True) -> None:
        checkpoint = torch.load(
            self.checkpoint_dir / f"{filename}.pth",
            map_location=self.device,
            weights_only=False,
        )
        self.vae.load_state_dict(checkpoint["vae_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.fixed_noise = checkpoint["fixed_noise"].to(self.device)
        self.start_epoch = int(checkpoint["epoch"]) + 1

        if restore_rng and (rng := checkpoint.get("rng_state")):
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
            if rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])

    def fit(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        project_name: str,
        group_name: str,
        run_name: str,
        *,
        sample_every: int = 1,
        checkpoint_every: int = 10,
        wandb_mode: str = "offline",
        wandb_config: dict[str, Any] | None = None,
    ) -> None:
        if sample_every < 1 or checkpoint_every < 1:
            raise ValueError("sample_every and checkpoint_every must be positive")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=project_name,
            group=group_name,
            name=run_name,
            mode=wandb_mode,
            config=wandb_config,
        )
        try:
            epoch_bar = tqdm(
                range(self.start_epoch, num_epochs),
                desc="VAE training",
            )
            for epoch in epoch_bar:
                metrics = self.train_one_epoch(dataloader)
                log_data: dict[str, Any] = {
                    "epoch": epoch + 1,
                    "train/reconstruction_loss": metrics["reconstruction_loss"],
                    "train/kl_loss": metrics["kl_loss"],
                    "train/total_loss": metrics["total_loss"],
                }

                if (epoch + 1) % sample_every == 0 or epoch == self.start_epoch:
                    sample_path = self.save_samples(epoch)
                    log_data["generated_samples"] = wandb.Image(str(sample_path))

                self.save_checkpoint("last", epoch)
                if (epoch + 1) % checkpoint_every == 0:
                    self.save_checkpoint(f"epoch_{epoch + 1:04d}", epoch)

                wandb.log(log_data, step=epoch + 1)
                epoch_bar.set_postfix(
                    recon=f"{metrics['reconstruction_loss']:.3f}",
                    kl=f"{metrics['kl_loss']:.4f}",
                )
        finally:
            run.finish()
