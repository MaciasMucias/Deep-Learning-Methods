import yaml

from pydantic import BaseModel
from pathlib import Path


class DataConfig(BaseModel):
    image_size: int


class TrainingConfig(BaseModel):
    lr: float
    batch_size: int
    num_epochs: int
    weight_decay: float
    num_workers: int


class ExperimentConfig(BaseModel):
    model_name: str
    data_root: Path
    checkpoint_dir: Path
    project_name: str
    run_name: str
    data: DataConfig
    training: TrainingConfig


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)
