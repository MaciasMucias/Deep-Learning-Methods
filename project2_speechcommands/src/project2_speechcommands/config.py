import yaml
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 160
    f_min: float = 0.0
    f_max: float = 8000.0
    target_length: int = 16000  # pad/trim audio to exactly 1 second of samples
    target_frames: int = 112  # pad spectrogram time axis to 112 = 7×16 patches


class TransformerConfig(BaseModel):
    patch_size: tuple[int, int] = (16, 16)
    embed_dim: int = 128  # swept: 128, 256
    n_heads: int = 4  # swept: 4, 8
    n_layers: int = 2  # swept: 2, 4
    mlp_ratio: float = 4.0


class BalanceConfig(BaseModel):
    strategy: Literal["none", "oversample", "prelim"] = "none"
    oversample_target_ratio: float = 0.5


class TrainingConfig(BaseModel):
    lr: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 60
    weight_decay: float = 1e-4
    num_workers: int = 4
    patience: int = 10


class ExperimentConfig(BaseModel):
    model_name: str
    num_classes: int = 12  # 12 for main model, 3 for prelim (known/unknown/silence)
    data_root: Path
    checkpoint_dir: Path
    project_name: str
    run_name: str
    audio: AudioConfig = Field(default_factory=AudioConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    balance: BalanceConfig = Field(default_factory=BalanceConfig)
    training: TrainingConfig


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)
