import random
import argparse

from collections import defaultdict
from torch.utils.data import Subset

from pathlib import Path

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from dl_base import Trainer, get_device, set_seed
from project1_cinic10.config import load_config
from project1_cinic10.data import build_transforms
from project1_cinic10.models import MODEL_REGISTRY


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to yaml config file")
    parser.add_argument("--seeds", type=int, nargs="+", help="seeds to use")
    parser.add_argument("--shots", type=int, help="number of samples per class")
    return parser.parse_args()


def get_few_shot_subset(dataset, shots: int, seed: int) -> Subset:
    class_to_indices = defaultdict(list)

    for idx, label in enumerate(dataset.targets):
        class_to_indices[label].append(idx)

    rng = random.Random(seed)
    selected_indices = []

    for label in sorted(class_to_indices.keys()):
        indices = class_to_indices[label]
        rng.shuffle(indices)
        selected_indices.extend(indices[:shots])

    return Subset(dataset, selected_indices)

def main() -> None:
    args = get_args()
    config = load_config(args.config)

    print(f"Running on {get_device().type}")

    for seed in args.seeds:
        set_seed(seed)
        device = get_device()
        root = Path(config.data_root)

        train_transforms, eval_transforms = build_transforms(
            mean=config.augmentation.mean,
            std=config.augmentation.std,
            horizontal_flip=config.augmentation.horizontal_flip,
            random_crop=config.augmentation.random_crop,
            rotation=config.augmentation.rotation,
            cutout=config.augmentation.cutout,
            crop_size=config.augmentation.crop_size,
            crop_padding=config.augmentation.crop_padding,
            rotation_range=config.augmentation.rotation_range,
            cutout_size=config.augmentation.cutout_size,
        )

        train_data = ImageFolder(root / "train", transform=train_transforms)
        val_data = ImageFolder(root / "valid", transform=eval_transforms)
        test_data = ImageFolder(root / "test", transform=eval_transforms)

        few_shot_train_data = get_few_shot_subset(train_data, args.shots, seed)

        pin_memory = device.type == "cuda"

        train_loader = DataLoader(
            few_shot_train_data,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=pin_memory,
        )

        model = MODEL_REGISTRY[config.model_name](dropout=config.training.dropout)
        model.to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        criterion = nn.CrossEntropyLoss()
        checkpoint_dir = config.checkpoint_dir / f"seed{seed}" / f"{args.shots}shot"
        trainer = Trainer(model, optimizer, criterion, device, checkpoint_dir)

        trainer.fit(
            train_loader,
            val_loader,
            config.training.num_epochs,
            config.project_name,
            f"{config.run_name}_{args.shots}shot_seed{seed}",
        )

if __name__ == "__main__":
    main()