import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from project3_catgen.generation import generate_images, sample_latent


class DCGANTrainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str | Path,
        latent_dim: int,
        sample_grid_size: int = 64,
        label_smoothing: float = 0.0,
    ) -> None:
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if sample_grid_size < 1:
            raise ValueError("sample_grid_size must be positive")

        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.latent_dim = latent_dim
        self.label_smoothing = label_smoothing
        self.criterion = nn.BCEWithLogitsLoss()
        self.fixed_noise = sample_latent(sample_grid_size, latent_dim, device)
        self.start_epoch = 0

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def train_one_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.generator.train()
        self.discriminator.train()

        totals = {
            "generator_loss": 0.0,
            "discriminator_loss": 0.0,
            "discriminator_real_score": 0.0,
            "discriminator_fake_score": 0.0,
        }
        total_samples = 0

        progress = tqdm(dataloader, desc="train", leave=False)
        for batch in progress:
            real_images = batch[0] if isinstance(batch, (list, tuple)) else batch
            real_images = real_images.to(self.device, non_blocking=True)
            batch_size = real_images.size(0)

            self._set_requires_grad(self.discriminator, True)
            self.discriminator_optimizer.zero_grad(set_to_none=True)

            real_logits = self.discriminator(real_images)
            real_targets = torch.full_like(real_logits, 1.0 - self.label_smoothing)
            discriminator_real_loss = self.criterion(real_logits, real_targets)

            noise = sample_latent(batch_size, self.latent_dim, self.device)
            fake_images = self.generator(noise)
            fake_logits = self.discriminator(fake_images.detach())
            discriminator_fake_loss = self.criterion(
                fake_logits, torch.zeros_like(fake_logits)
            )
            discriminator_loss = (
                discriminator_real_loss + discriminator_fake_loss
            ) / 2.0
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            self._set_requires_grad(self.discriminator, False)
            self.generator_optimizer.zero_grad(set_to_none=True)
            generator_logits = self.discriminator(fake_images)
            generator_loss = self.criterion(
                generator_logits, torch.ones_like(generator_logits)
            )
            generator_loss.backward()
            self.generator_optimizer.step()
            self._set_requires_grad(self.discriminator, True)

            totals["generator_loss"] += generator_loss.item() * batch_size
            totals["discriminator_loss"] += discriminator_loss.item() * batch_size
            totals["discriminator_real_score"] += (
                real_logits.sigmoid().mean().item() * batch_size
            )
            totals["discriminator_fake_score"] += (
                fake_logits.sigmoid().mean().item() * batch_size
            )
            total_samples += batch_size

            progress.set_postfix(
                g=f"{generator_loss.item():.3f}",
                d=f"{discriminator_loss.item():.3f}",
            )

        if total_samples == 0:
            raise ValueError("Training dataloader is empty")
        return {name: value / total_samples for name, value in totals.items()}

    def save_samples(self, epoch: int) -> Path:
        images = generate_images(
            self.generator,
            self.fixed_noise,
            batch_size=self.fixed_noise.size(0),
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
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "generator_optimizer_state_dict": self.generator_optimizer.state_dict(),
                "discriminator_optimizer_state_dict": (
                    self.discriminator_optimizer.state_dict()
                ),
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
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.generator_optimizer.load_state_dict(
            checkpoint["generator_optimizer_state_dict"]
        )
        self.discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer_state_dict"]
        )
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
                desc="DCGAN training",
            )
            for epoch in epoch_bar:
                metrics = self.train_one_epoch(dataloader)
                log_data: dict[str, Any] = {
                    "epoch": epoch + 1,
                    "train/generator_loss": metrics["generator_loss"],
                    "train/discriminator_loss": metrics["discriminator_loss"],
                    "train/discriminator_real_score": metrics[
                        "discriminator_real_score"
                    ],
                    "train/discriminator_fake_score": metrics[
                        "discriminator_fake_score"
                    ],
                }

                if (epoch + 1) % sample_every == 0 or epoch == self.start_epoch:
                    sample_path = self.save_samples(epoch)
                    log_data["generated_samples"] = wandb.Image(str(sample_path))

                self.save_checkpoint("last", epoch)
                if (epoch + 1) % checkpoint_every == 0:
                    self.save_checkpoint(f"epoch_{epoch + 1:04d}", epoch)

                wandb.log(log_data, step=epoch + 1)
                epoch_bar.set_postfix(
                    g=f"{metrics['generator_loss']:.3f}",
                    d=f"{metrics['discriminator_loss']:.3f}",
                )
        finally:
            run.finish()
