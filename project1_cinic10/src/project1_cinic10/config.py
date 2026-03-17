import yaml

from pydantic import BaseModel
from pathlib import Path

class AugmentationConfig(BaseModel):
    horizontal_flip: bool
    random_crop: bool
    rotation: bool
    cutout: bool
    crop_size: int
    crop_padding: int
    rotation_range: int
    cutout_size: int

class TrainingConfig(BaseModel):
    lr: float
    batch_size: int
    num_epochs: int
    weight_decay: float
    dropout: float
    num_workers: int

class ExperimentConfig(BaseModel):
    model_name: str
    data_root: Path
    checkpoint_dir: Path
    project_name: str
    run_name: str
    augmentation: AugmentationConfig
    training: TrainingConfig


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)