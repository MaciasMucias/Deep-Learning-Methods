import math

import torch
import torch.nn as nn


def _validate_image_size(image_size: int) -> None:
    if image_size < 16 or image_size & (image_size - 1):
        raise ValueError("image_size must be a power of two and at least 16")


class Generator(nn.Module):
    """DCGAN generator producing RGB images in the [-1, 1] range."""

    def __init__(
        self,
        latent_dim: int = 128,
        feature_maps: int = 64,
        image_size: int = 64,
        image_channels: int = 3,
    ) -> None:
        super().__init__()
        _validate_image_size(image_size)

        num_upsamples = int(math.log2(image_size)) - 2
        current_multiplier = min(8, 2 ** (num_upsamples - 1))
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                latent_dim,
                feature_maps * current_multiplier,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * current_multiplier),
            nn.ReLU(inplace=True),
        ]

        for step in range(num_upsamples - 1):
            next_multiplier = min(8, 2 ** (num_upsamples - 2 - step))
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        feature_maps * current_multiplier,
                        feature_maps * next_multiplier,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(feature_maps * next_multiplier),
                    nn.ReLU(inplace=True),
                ]
            )
            current_multiplier = next_multiplier

        layers.extend(
            [
                nn.ConvTranspose2d(
                    feature_maps * current_multiplier,
                    image_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.Tanh(),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class Discriminator(nn.Module):
    """DCGAN discriminator returning one unnormalised logit per image."""

    def __init__(
        self,
        feature_maps: int = 64,
        image_size: int = 64,
        image_channels: int = 3,
    ) -> None:
        super().__init__()
        _validate_image_size(image_size)

        num_downsamples = int(math.log2(image_size)) - 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                image_channels,
                feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        current_multiplier = 1
        for step in range(1, num_downsamples):
            next_multiplier = min(8, 2**step)
            layers.extend(
                [
                    nn.Conv2d(
                        feature_maps * current_multiplier,
                        feature_maps * next_multiplier,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(feature_maps * next_multiplier),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current_multiplier = next_multiplier

        layers.append(
            nn.Conv2d(
                feature_maps * current_multiplier,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images).flatten()


def initialize_dcgan_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)
