"""Generate a linear interpolation between two DCGAN latent vectors."""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from dl_base import get_device, set_seed
from project3_catgen.config import load_config
from project3_catgen.experiments.utils import build_dcgan
from project3_catgen.generation import (
    generate_images,
    interpolate_latents,
    sample_latent,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--checkpoint", type=str, default="last")
    parser.add_argument("--latent-seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    config = load_config(args.config)
    device = get_device()
    set_seed(args.latent_seed)

    generator, _ = build_dcgan(config)
    generator.to(device)
    checkpoint_dir = config.checkpoint_dir / f"{config.run_name}_seed({args.seed})"
    checkpoint_path = checkpoint_dir / f"{args.checkpoint}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])

    endpoints = sample_latent(2, config.dcgan.latent_dim, device)
    latents, alphas = interpolate_latents(endpoints[0], endpoints[1], args.steps)
    images = generate_images(generator, latents, batch_size=args.steps)

    output_path = args.output or checkpoint_dir / "interpolation.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        images,
        output_path,
        nrow=args.steps,
        normalize=True,
        value_range=(-1, 1),
    )
    torch.save(
        {
            "z1": endpoints[0].cpu(),
            "z2": endpoints[1].cpu(),
            "alphas": alphas.cpu(),
            "latents": latents.cpu(),
            "checkpoint": str(checkpoint_path),
            "latent_seed": args.latent_seed,
        },
        output_path.with_suffix(".pt"),
    )
    print(f"Saved interpolation grid to {output_path}")
    print(f"Saved latent vectors to {output_path.with_suffix('.pt')}")


if __name__ == "__main__":
    main()
