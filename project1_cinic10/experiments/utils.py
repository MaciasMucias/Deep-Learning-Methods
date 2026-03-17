from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dl_base import set_seed, get_device, Trainer
from project1_cinic10.config import ExperimentConfig
from project1_cinic10.data import get_dataloaders
from project1_cinic10.models import MODEL_REGISTRY


def setup_experiment(config: ExperimentConfig, seed: int) -> tuple[Trainer, DataLoader, DataLoader, DataLoader]:
    device = get_device()
    set_seed(seed)
    checkpoint_dir = config.checkpoint_dir / f"seed{seed}"
    train_loader, val_loader, test_loader = get_dataloaders(
        config.data_root,
        config.augmentation,
        config.training.batch_size,
        config.training.num_workers
    )
    model = MODEL_REGISTRY[config.model_name](dropout=config.training.dropout)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device, checkpoint_dir)
    return trainer, train_loader, val_loader, test_loader