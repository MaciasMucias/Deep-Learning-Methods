import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from pathlib import Path

from dl_base.src.dl_base import get_device


class Cutout:
    __slots__ = ('size', 'top_left_offset', 'bottom_right_offset')
    def __init__(self, size: int = 16) -> None:
        self.size = size
        self.top_left_offset = size//2
        self.bottom_right_offset = size - self.top_left_offset

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        if h < self.size: raise ValueError(f'Image too short ({h} < {self.size})')
        if w < self.size: raise ValueError(f'Image too narrow ({w} < {self.size})')
        x = torch.randint(0, w, (1,)).item()
        y = torch.randint(0, h, (1,)).item()

        x_start = max(0, x - self.top_left_offset)
        x_end   = min(w, x + self.bottom_right_offset)

        y_start = max(0, y - self.top_left_offset)
        y_end   = min(h, y + self.bottom_right_offset)

        img[:, y_start:y_end, x_start:x_end] = 0
        return img

    def __repr__(self) -> str:
        return f"Cutout(size={self.size})"


def build_transforms(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    horizontal_flip: bool = False,
    random_crop: bool = False,
    rotation: bool = False,
    cutout: bool = False,
    crop_size: int = 32,
    crop_padding: int = 4,
    rotation_range: int = 15,
    cutout_size: int = 16,
) -> tuple[transforms.Compose, transforms.Compose]:

    transforms_list = []
    if horizontal_flip:
        transforms_list.append(transforms.RandomHorizontalFlip())
    if random_crop:
        transforms_list.append(transforms.RandomCrop(size=crop_size, padding=crop_padding))
    if rotation:
        transforms_list.append(transforms.RandomRotation(rotation_range))

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean, std))

    if cutout:
        transforms_list.append(Cutout(size=cutout_size))

    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transforms.Compose(transforms_list), eval_transforms




def get_dataloaders(
    root: str | Path,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    horizontal_flip: bool = False,
    random_crop: bool = False,
    rotation: bool = False,
    cutout: bool = False,
    crop_size: int = 32,
    crop_padding: int = 4,
    rotation_range: int = 15,
    cutout_size: int = 16,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(root)

    train_transforms, eval_transforms = build_transforms(mean=mean, std=std, horizontal_flip=horizontal_flip, random_crop=random_crop, rotation=rotation, cutout=cutout, crop_size=crop_size, crop_padding=crop_padding, rotation_range=rotation_range, cutout_size=cutout_size)

    train_data  = ImageFolder(root / "train",   transform=train_transforms)
    val_data    = ImageFolder(root / "valid",   transform=eval_transforms)
    test_data   = ImageFolder(root / "test",    transform=eval_transforms)

    pin_memory = get_device().type == "cuda"

    train_loader    = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader      = DataLoader(val_data,   batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    test_loader     = DataLoader(test_data,  batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
