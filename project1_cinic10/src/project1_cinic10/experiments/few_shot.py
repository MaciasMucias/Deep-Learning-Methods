import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision.datasets import ImageFolder

from dl_base import get_device, set_seed
from project1_cinic10.config import load_config
from project1_cinic10.data import build_transforms


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to yaml config file")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="seeds to use")
    parser.add_argument("--shots", type=int, required=True, help="number of support samples per class")
    parser.add_argument("--ways", type=int, default=10, help="number of classes per episode")
    parser.add_argument("--queries", type=int, default=5, help="number of query samples per class")
    parser.add_argument("--episodes-per-epoch", type=int, default=100, help="training episodes per epoch")
    parser.add_argument("--eval-episodes", type=int, default=100, help="validation/test episodes")
    return parser.parse_args()


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
    """
    Small CNN backbone for ProtoNet.
    Input: 3x32x32
    Output: embedding vector
    """
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),    # 32 -> 16
            ConvBlock(64, 64),   # 16 -> 8
            ConvBlock(64, 64),   # 8 -> 4
            ConvBlock(64, 64),   # 4 -> 2
        )
        self.proj = nn.Linear(64 * 2 * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        return x


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
        if n_way > len(self.classes):
            raise ValueError(f"Requested n_way={n_way}, but dataset has only {len(self.classes)} classes.")

        episode_classes = rng.sample(self.classes, n_way)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for episode_label, original_class in enumerate(episode_classes):
            indices = self.class_to_indices[original_class]
            needed = n_shot + n_query

            if len(indices) < needed:
                raise ValueError(
                    f"Class {original_class} has only {len(indices)} samples, "
                    f"but episode needs {needed}."
                )

            selected = rng.sample(indices, needed)
            support_idx = selected[:n_shot]
            query_idx = selected[n_shot:]

            for idx in support_idx:
                image, _ = self.dataset[idx]
                support_images.append(image)
                support_labels.append(episode_label)

            for idx in query_idx:
                image, _ = self.dataset[idx]
                query_images.append(image)
                query_labels.append(episode_label)

        support_x = torch.stack(support_images).to(device)
        support_y = torch.tensor(support_labels, dtype=torch.long, device=device)
        query_x = torch.stack(query_images).to(device)
        query_y = torch.tensor(query_labels, dtype=torch.long, device=device)

        return support_x, support_y, query_x, query_y


def compute_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, n_way: int) -> torch.Tensor:
    prototypes = []
    for c in range(n_way):
        class_embeddings = embeddings[labels == c]
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes, dim=0)


def prototypical_loss(
    encoder: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    n_way: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    support_embeddings = encoder(support_x)  # [n_way*n_shot, emb_dim]
    query_embeddings = encoder(query_x)      # [n_way*n_query, emb_dim]

    prototypes = compute_prototypes(support_embeddings, support_y, n_way)  # [n_way, emb_dim]

    distances = torch.cdist(query_embeddings, prototypes, p=2)  # [num_query, n_way]
    logits = -distances

    loss = F.cross_entropy(logits, query_y)
    preds = logits.argmax(dim=1)
    acc = (preds == query_y).float().mean()

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

    losses = []
    accs = []

    for _ in range(eval_episodes):
        support_x, support_y, query_x, query_y = episodic_index.sample_episode(
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            rng=rng,
            device=device,
        )

        loss, acc = prototypical_loss(
            encoder=encoder,
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            n_way=n_way,
        )

        losses.append(loss.item())
        accs.append(acc.item())

    return float(sum(losses) / len(losses)), float(sum(accs) / len(accs))


def train_one_seed(args: argparse.Namespace, config, seed: int) -> None:
    set_seed(seed)
    device = get_device()
    root = Path(config.data_root)

    mean = getattr(config.augmentation, "mean", [0.4789, 0.4723, 0.4305])
    std = getattr(config.augmentation, "std", [0.2421, 0.2383, 0.2587])

    train_transforms, eval_transforms = build_transforms(
        mean=mean,
        std=std,
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

    train_episodic = EpisodicIndex(train_data)
    val_episodic = EpisodicIndex(val_data)
    test_episodic = EpisodicIndex(test_data)

    encoder = ProtoNetEncoder(embedding_dim=128).to(device)

    optimizer = AdamW(
        encoder.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    checkpoint_dir = Path(config.checkpoint_dir) / "few_shot_protonet" / f"seed{seed}" / f"{args.shots}shot"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best.pt"

    best_val_acc = -1.0

    print(f"\n=== Seed {seed} ===")
    print(f"Device: {device.type}")
    print(
        f"Few-shot setup: {args.ways}-way, {args.shots}-shot, "
        f"{args.queries} query/class, {args.episodes_per_epoch} train episodes/epoch"
    )

    for epoch in range(1, config.training.num_epochs + 1):
        encoder.train()
        rng = random.Random(seed * 10_000 + epoch)

        train_losses = []
        train_accs = []

        for _ in range(args.episodes_per_epoch):
            support_x, support_y, query_x, query_y = train_episodic.sample_episode(
                n_way=args.ways,
                n_shot=args.shots,
                n_query=args.queries,
                rng=rng,
                device=device,
            )

            loss, acc = prototypical_loss(
                encoder=encoder,
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                query_y=query_y,
                n_way=args.ways,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(acc.item())

        train_loss = float(sum(train_losses) / len(train_losses))
        train_acc = float(sum(train_accs) / len(train_accs))

        val_loss, val_acc = evaluate(
            encoder=encoder,
            episodic_index=val_episodic,
            n_way=args.ways,
            n_shot=args.shots,
            n_query=args.queries,
            eval_episodes=args.eval_episodes,
            device=device,
            seed=seed + epoch,
        )

        print(
            f"Epoch {epoch:03d}/{config.training.num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                },
                best_ckpt_path,
            )

    print(f"\nBest val_acc for seed {seed}: {best_val_acc:.4f}")

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(
        encoder=encoder,
        episodic_index=test_episodic,
        n_way=args.ways,
        n_shot=args.shots,
        n_query=args.queries,
        eval_episodes=args.eval_episodes,
        device=device,
        seed=seed + 9999,
    )

    print(f"Test results | loss={test_loss:.4f} acc={test_acc:.4f}")


def main() -> None:
    args = get_args()
    config = load_config(args.config)

    print(f"Running ProtoNet few-shot on {get_device().type}")

    for seed in args.seeds:
        train_one_seed(args, config, seed)


if __name__ == "__main__":
    main()