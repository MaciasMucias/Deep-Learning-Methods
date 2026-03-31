import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

from pathlib import Path

from tqdm import tqdm

from dl_base import get_device
from project1_cinic10.config import AugmentationConfig

CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC10_STD  = (0.24205776, 0.23828046, 0.25874835)
CINIC10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


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
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean, std))

    # All augmentations after ToTensor — will be applied lazily at __getitem__
    if horizontal_flip:
        transforms_list.append(transforms.RandomHorizontalFlip())
    if random_crop:
        transforms_list.append(transforms.RandomCrop(size=crop_size, padding=crop_padding))
    if rotation:
        transforms_list.append(transforms.RandomRotation(rotation_range))
    if cutout:
        transforms_list.append(Cutout(size=cutout_size))

    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transforms.Compose(transforms_list), eval_transforms


class PreloadedDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose, batch_size: int, num_workers: int, persistent_workers: bool) -> None:
        # Split transform: everything up to and including ToTensor for preload, everything after for runtime
        to_tensor_idx = next(i for i, t in enumerate(transform.transforms)
                             if isinstance(t, transforms.ToTensor))

        preload_transform = transforms.Compose(transform.transforms[:to_tensor_idx + 1])
        self.transform = transforms.Compose(transform.transforms[to_tensor_idx + 1:]) \
            if transform.transforms[to_tensor_idx + 1:] else None

        dataset = ImageFolder(root, transform=preload_transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers)

        images, labels = [], []
        for imgs, lbls in tqdm(loader, desc=f"preloading {root.name}"):
            images.append(imgs)
            labels.append(lbls)

        self.images = torch.cat(images)  # stored as float tensors, normalized
        self.labels = torch.cat(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]



def get_dataloaders(
    root: str | Path,
    augmentation: AugmentationConfig,
    batch_size: int = 64,
    num_workers: int = 4,
    test_mode: bool = False,
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None]:
    root = Path(root)

    train_transforms, eval_transforms = build_transforms(
        mean            = CINIC10_MEAN,
        std             = CINIC10_STD,
        horizontal_flip = augmentation.horizontal_flip,
        random_crop     = augmentation.random_crop,
        rotation        = augmentation.rotation,
        cutout          = augmentation.cutout,
        crop_size       = augmentation.crop_size,
        crop_padding    = augmentation.crop_padding,
        rotation_range  = augmentation.rotation_range,
        cutout_size     = augmentation.cutout_size)

    pin_memory = get_device().type == "cuda"

    if not test_mode:
        train_data      = PreloadedDataset(root / "train",   transform=train_transforms, batch_size=batch_size, num_workers=num_workers, persistent_workers=num_workers > 0)
        val_data        = PreloadedDataset(root / "valid",   transform=eval_transforms, batch_size=batch_size, num_workers=num_workers, persistent_workers=num_workers > 0)
        train_loader    = DataLoader(train_data,             batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, persistent_workers=num_workers > 0)
        val_loader      = DataLoader(val_data,               batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, val_loader, None

    test_data   = PreloadedDataset(root / "test",    transform=eval_transforms, batch_size=batch_size, num_workers=num_workers, persistent_workers=num_workers > 0)
    test_loader     = DataLoader(test_data,  batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return None, None, test_loader

