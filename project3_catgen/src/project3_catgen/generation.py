from collections.abc import Callable, Iterator

import torch
import torch.nn as nn


def sample_latent(
    num_samples: int,
    latent_dim: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return torch.randn(
        num_samples,
        latent_dim,
        1,
        1,
        device=device,
        generator=generator,
    )


def sample_latent_vae(
    num_samples: int,
    latent_dim: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return torch.randn(num_samples, latent_dim, device=device, generator=generator)


def interpolate_latents(
    start: torch.Tensor,
    end: torch.Tensor,
    steps: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    if start.shape != end.shape:
        raise ValueError("start and end latent vectors must have identical shapes")
    if steps < 2:
        raise ValueError("steps must be at least 2")

    alphas = torch.linspace(0.0, 1.0, steps, device=start.device)
    view_shape = (steps,) + (1,) * start.ndim
    interpolated = (
        (1.0 - alphas.view(view_shape)) * start.unsqueeze(0)
        + alphas.view(view_shape) * end.unsqueeze(0)
    )
    return interpolated, alphas


def _batches(tensor: torch.Tensor, batch_size: int) -> Iterator[torch.Tensor]:
    for start in range(0, tensor.size(0), batch_size):
        yield tensor[start : start + batch_size]


@torch.no_grad()
def generate_images(
    generator: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
    latents: torch.Tensor,
    batch_size: int = 64,
    *,
    module: nn.Module | None = None,
) -> torch.Tensor:
    mod = module if module is not None else generator
    was_training = mod.training if isinstance(mod, nn.Module) else False
    if isinstance(mod, nn.Module):
        mod.eval()
    images = [generator(batch).cpu() for batch in _batches(latents, batch_size)]
    if isinstance(mod, nn.Module):
        mod.train(was_training)
    return torch.cat(images, dim=0)
