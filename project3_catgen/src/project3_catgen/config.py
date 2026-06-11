import yaml

from pydantic import BaseModel, Field
from pathlib import Path


class DataConfig(BaseModel):
    image_size: int = 64


class TrainingConfig(BaseModel):
    lr: float = 2e-4
    batch_size: int = 64
    num_epochs: int = 100
    weight_decay: float = 0.0
    num_workers: int = 0
    sample_every: int = 1
    checkpoint_every: int = 10


class DCGANConfig(BaseModel):
    latent_dim: int = 128
    generator_features: int = 64
    discriminator_features: int = 64
    beta1: float = 0.5
    beta2: float = 0.999
    label_smoothing: float = 0.0
    sample_grid_size: int = 64


class VAEConfig(BaseModel):
    latent_dim: int = 128
    encoder_features: int = 64
    decoder_features: int = 64
    beta: float = 1.0
    sample_grid_size: int = 64


class ExperimentConfig(BaseModel):
    model_name: str
    data_root: Path
    checkpoint_dir: Path
    project_name: str
    run_name: str
    data: DataConfig
    training: TrainingConfig
    dcgan: DCGANConfig = Field(default_factory=DCGANConfig)
    vae: VAEConfig = Field(default_factory=VAEConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)
