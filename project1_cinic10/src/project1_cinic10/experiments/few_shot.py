import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.optim import AdamW
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

from dl_base import get_device, set_seed

# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = Path("project1_cinic10/data")
CHECKPOINT_ROOT = Path("project1_cinic10/runs")

PROJECT_NAME = "deep-learning-cinic10"
RUN_GROUP = "fewshot"

WAYS = 5
QUERIES = 5
SHOTS_LIST = [1, 5, 10]
SEEDS = [0, 1, 42]

LR = 0.001
NUM_EPOCHS = 75
WEIGHT_DECAY = 0.0001
DROPOUT = 0.3
EMBEDDING_DIM = 128
EPISODES_PER_EPOCH = 100
EVAL_EPISODES = 100
PATIENCE = 10

MEAN = [0.4789, 0.4723, 0.4305]
STD = [0.2421, 0.2383, 0.2587]

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ProtoNetEncoder(nn.Module):
    """4-block CNN backbone. Input: 3×32×32 → embedding vector."""

    def __init__(self, embedding_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(64 * 2 * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.proj(x)


class EpisodicIndex:
    def __init__(self, dataset: ImageFolder) -> None:
        self.dataset = dataset
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(dataset.targets):
            self.class_to_indices[label].append(idx)
        self.classes = sorted(self.class_to_indices.keys())

    def sample_episode(
        self,
        n_way: int,
        n_shot: int,
        n_query: int,
        rng: random.Random,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        episode_classes = rng.sample(self.classes, n_way)

        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for ep_label, cls in enumerate(episode_classes):
            indices = self.class_to_indices[cls]
            selected = rng.sample(indices, n_shot + n_query)

            for idx in selected[:n_shot]:
                support_images.append(self.dataset[idx][0])
                support_labels.append(ep_label)

            for idx in selected[n_shot:]:
                query_images.append(self.dataset[idx][0])
                query_labels.append(ep_label)

        return (
            torch.stack(support_images).to(device),
            torch.tensor(support_labels, dtype=torch.long, device=device),
            torch.stack(query_images).to(device),
            torch.tensor(query_labels, dtype=torch.long, device=device),
        )


def compute_prototypes(
    embeddings: torch.Tensor, labels: torch.Tensor, n_way: int
) -> torch.Tensor:
    return torch.stack([embeddings[labels == c].mean(dim=0) for c in range(n_way)])


def prototypical_loss(
    encoder: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    n_way: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prototypes = compute_prototypes(encoder(support_x), support_y, n_way)
    logits = -torch.cdist(encoder(query_x), prototypes, p=2)
    loss = F.cross_entropy(logits, query_y)
    acc = (logits.argmax(dim=1) == query_y).float().mean()
    return loss, acc


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    episodic_index: EpisodicIndex,
    n_way: int,
    n_shot: int,
    n_query: int,
    eval_episodes: int,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    encoder.eval()
    rng = random.Random(seed)
    losses, accs = [], []

    for _ in range(eval_episodes):
        support_x, support_y, query_x, query_y = episodic_index.sample_episode(
            n_way, n_shot, n_query, rng, device
        )
        loss, acc = prototypical_loss(
            encoder, support_x, support_y, query_x, query_y, n_way
        )
        losses.append(loss.item())
        accs.append(acc.item())

    return sum(losses) / len(losses), sum(accs) / len(accs)


def save_checkpoint(
    path: Path,
    epoch: int,
    encoder: nn.Module,
    optimizer: AdamW,
    best_val_acc: float,
    seed: int,
    shots: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "seed": seed,
            "shots": shots,
            "ways": WAYS,
            "queries": QUERIES,
            "lr": LR,
            "num_epochs": NUM_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "embedding_dim": EMBEDDING_DIM,
            "episodes_per_epoch": EPISODES_PER_EPOCH,
            "eval_episodes": EVAL_EPISODES,
        },
        path,
    )


def train_one_run(seed: int, shots: int) -> None:
    set_seed(seed)
    device = get_device()

    train_data = ImageFolder(DATA_ROOT / "train", transform=TRANSFORM)
    val_data = ImageFolder(DATA_ROOT / "valid", transform=TRANSFORM)
    test_data = ImageFolder(DATA_ROOT / "test", transform=TRANSFORM)

    train_episodic = EpisodicIndex(train_data)
    val_episodic = EpisodicIndex(val_data)
    test_episodic = EpisodicIndex(test_data)

    encoder = ProtoNetEncoder(EMBEDDING_DIM, DROPOUT).to(device)
    optimizer = AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    checkpoint_dir = CHECKPOINT_ROOT / "fewshot" / f"{shots}shot" / f"seed{seed}"

    run_name = f"fewshot_{shots}shot_seed{seed}"

    wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        group=RUN_GROUP,
        config={
            "seed": seed,
            "shots": shots,
            "ways": WAYS,
            "queries": QUERIES,
            "episodes_per_epoch": EPISODES_PER_EPOCH,
            "eval_episodes": EVAL_EPISODES,
            "lr": LR,
            "num_epochs": NUM_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "embedding_dim": EMBEDDING_DIM,
            "augmentation": "none",
            "mean": MEAN,
            "std": STD,
        },
    )

    best_val_acc = -1.0
    epochs_without_improvement = 0

    print(f"\nSeed {seed} | {shots}-shot")
    epoch_bar = tqdm(range(NUM_EPOCHS), desc=run_name)

    for epoch in epoch_bar:
        encoder.train()
        rng = random.Random(seed * 10_000 + epoch)
        train_losses, train_accs = [], []

        for _ in range(EPISODES_PER_EPOCH):
            support_x, support_y, query_x, query_y = train_episodic.sample_episode(
                WAYS, shots, QUERIES, rng, device
            )
            loss, acc = prototypical_loss(
                encoder, support_x, support_y, query_x, query_y, WAYS
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accs.append(acc.item())

        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)

        val_loss, val_acc = evaluate(
            encoder,
            val_episodic,
            WAYS,
            shots,
            QUERIES,
            EVAL_EPISODES,
            device,
            seed=seed + epoch + 1,
        )

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
            }
        )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            epoch,
            encoder,
            optimizer,
            best_val_acc,
            seed,
            shots,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_dir / "best.pt",
                epoch,
                encoder,
                optimizer,
                best_val_acc,
                seed,
                shots,
            )
        else:
            epochs_without_improvement += 1

        epoch_bar.set_postfix(
            val_acc=f"{val_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
            patience=f"{epochs_without_improvement}/{PATIENCE}",
        )

        if epochs_without_improvement >= PATIENCE:
            tqdm.write(f"Early stopping at epoch {epoch}.")
            break

    best_ckpt = torch.load(checkpoint_dir / "best.pt", map_location=device)
    encoder.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc = evaluate(
        encoder,
        test_episodic,
        WAYS,
        shots,
        QUERIES,
        EVAL_EPISODES,
        device,
        seed=seed + 9999,
    )

    wandb.log(
        {
            "best_val_acc": best_val_acc,
            "test/loss": test_loss,
            "test/acc": test_acc,
        }
    )

    print(
        f"Done {run_name} | best_val_acc={best_val_acc:.4f} "
        f"| test_loss={test_loss:.4f} | test_acc={test_acc:.4f}"
    )
    wandb.finish()


def main() -> None:
    print(f"Running fewshot on {get_device().type}")
    print(f"Planned runs: {len(SEEDS) * len(SHOTS_LIST)}")
    for shots in SHOTS_LIST:
        for seed in SEEDS:
            train_one_run(seed=seed, shots=shots)


if __name__ == "__main__":
    main()
