import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dl_base import get_device, set_seed
from project3_catgen.config import ExperimentConfig
from project3_catgen.data import get_dataloaders
from project3_catgen.dcgan_trainer import DCGANTrainer
from project3_catgen.models import (
    Discriminator,
    Generator,
    initialize_dcgan_weights,
)


def build_dcgan(
    config: ExperimentConfig,
) -> tuple[Generator, Discriminator]:
    if config.model_name != "dcgan":
        raise ValueError(f"Expected model_name='dcgan', got {config.model_name!r}")

    generator = Generator(
        latent_dim=config.dcgan.latent_dim,
        feature_maps=config.dcgan.generator_features,
        image_size=config.data.image_size,
    )
    discriminator = Discriminator(
        feature_maps=config.dcgan.discriminator_features,
        image_size=config.data.image_size,
    )
    generator.apply(initialize_dcgan_weights)
    discriminator.apply(initialize_dcgan_weights)
    return generator, discriminator


def setup_dcgan_experiment(
    config: ExperimentConfig,
    seed: int,
) -> tuple[DCGANTrainer, DataLoader]:
    device = get_device()
    set_seed(seed)
    checkpoint_dir = config.checkpoint_dir / f"{config.run_name}_seed({seed})"

    train_loader, _, _ = get_dataloaders(
        config.data_root,
        config.data,
        config.training.batch_size,
        config.training.num_workers,
        test_mode=False,
    )
    if train_loader is None:
        raise RuntimeError("Training dataloader was not created")

    generator, discriminator = build_dcgan(config)
    generator.to(device)
    discriminator.to(device)

    optimizer_kwargs = {
        "lr": config.training.lr,
        "betas": (config.dcgan.beta1, config.dcgan.beta2),
        "weight_decay": config.training.weight_decay,
    }
    generator_optimizer = Adam(generator.parameters(), **optimizer_kwargs)
    discriminator_optimizer = Adam(discriminator.parameters(), **optimizer_kwargs)

    trainer = DCGANTrainer(
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        device,
        checkpoint_dir,
        config.dcgan.latent_dim,
        config.dcgan.sample_grid_size,
        config.dcgan.label_smoothing,
    )
    return trainer, train_loader
