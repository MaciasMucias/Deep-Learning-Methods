import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dl_base import set_seed, get_device, Trainer
from project2_speechcommands.config import ExperimentConfig
from project2_speechcommands.data import get_dataloaders
from project2_speechcommands.models import MODEL_REGISTRY


def build_model(config: ExperimentConfig) -> nn.Module:
    kwargs: dict = {"num_classes": config.num_classes}

    transformer_kwargs = config.transformer.model_dump()

    if config.model_name == "transformer":
        kwargs |= transformer_kwargs

    elif config.model_name == "cnn_transformer":
        transformer_kwargs.pop("patch_size")
        kwargs |= transformer_kwargs

    return MODEL_REGISTRY[config.model_name](**kwargs)


def setup_experiment(
    config: ExperimentConfig,
    seed: int,
    *,
    test_mode: bool = False,
) -> tuple[Trainer, DataLoader, DataLoader, DataLoader]:
    device = get_device()
    set_seed(seed)
    checkpoint_dir = config.checkpoint_dir / f"{config.run_name}_seed({seed})"

    train_loader, val_loader, test_loader = get_dataloaders(
        config.data_root,
        config.audio,
        config.balance,
        config.training.batch_size,
        config.training.num_workers,
        test_mode=test_mode,
        remap_prelim=(config.balance.strategy == "prelim"),
    )

    model = build_model(config)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device,
        checkpoint_dir,
        patience=config.training.patience,
    )
    return trainer, train_loader, val_loader, test_loader
