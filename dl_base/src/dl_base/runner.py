import torch
from torch import nn
from torch.utils.data import DataLoader

import wandb
from pathlib import Path
from tqdm import tqdm


class Trainer:

    __slots__ = ("model", "optimizer", "criterion", "device", "checkpoint_dir", "best_val_acc", "start_epoch")

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, checkpoint_dir: str | Path) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_val_acc: float = 0.0
        self.start_epoch: int = 0

    def train_one_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        total_samples: int = 0
        total_correct: int = 0
        total_loss: float = 0.0

        self.model.train()
        for batch in tqdm(dataloader, desc="train", leave=False):
            self.optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == targets).sum().item()
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)


        return total_loss / total_samples, total_correct / total_samples

    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        total_samples: int = 0
        total_correct: int = 0
        total_loss: float = 0.0

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="eval", leave=False):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == targets).sum().item()
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def _save_checkpoint(self, filename: str, epoch: int) -> None:
        torch.save(
            {
                "start_epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
            }, self.checkpoint_dir / f"{filename}.pth")

    def load_checkpoint(self, filename: str) -> None:
        checkpoint = torch.load(self.checkpoint_dir / f"{filename}.pth", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['start_epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, project_name: str, run_name: str) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(project=project_name, name=run_name)

        epoch_bar = tqdm(range(self.start_epoch, num_epochs), desc="Training")
        for epoch in epoch_bar:
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
            })

            self._save_checkpoint("last", epoch)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint("best", epoch)

            epoch_bar.set_postfix(val_acc=f"{val_acc:.4f}", val_loss=f"{val_loss:.4f}")
        wandb.finish()