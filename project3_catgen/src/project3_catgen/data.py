import random
import warnings

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm

from dl_base import get_device
from project3_catgen.config import DataConfig

CAT_MEAN = (0.5, 0.5, 0.5)
CAT_STD = (0.5, 0.5, 0.5)
FID_SPLIT_SEED = 42
FID_SPLIT_RATIO = 0.1


class PreloadedCatDataset(Dataset):
    def __init__(self, paths: list[Path], transform: transforms.Compose) -> None:
        images = []
        for path in tqdm(paths, desc="preloading images"):
            try:
                with Image.open(path) as img:
                    images.append(transform(img.convert("RGB")))
            except Exception as e:
                warnings.warn(f"Skipping corrupt file {path}: {e}")
        if not images:
            raise ValueError(
                f"No valid images were loaded (n_paths={len(paths)}). Check the dataset path and file integrity."
            )
        self.images = torch.stack(images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[idx]


def get_dataloaders(
    root: str | Path,
    data_config: DataConfig,
    batch_size: int = 64,
    num_workers: int = 0,
    test_mode: bool = False,
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None]:
    root = Path(root)
    all_paths = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    rng = random.Random(FID_SPLIT_SEED)
    shuffled = all_paths.copy()
    rng.shuffle(shuffled)
    split = int(len(shuffled) * FID_SPLIT_RATIO)
    fid_paths = shuffled[:split]
    train_paths = shuffled[split:]

    transform = transforms.Compose([
        transforms.Resize((data_config.image_size, data_config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CAT_MEAN, CAT_STD),
    ])

    pin_memory = get_device().type == "cuda"

    if not test_mode:
        train_dataset = PreloadedCatDataset(train_paths, transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        return train_loader, None, None

    fid_dataset = PreloadedCatDataset(fid_paths, transform)
    fid_loader = DataLoader(
        fid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return None, None, fid_loader
