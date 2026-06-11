import math

import torch
import torch.nn as nn


def _validate_image_size(image_size: int) -> None:
    if image_size < 16 or image_size & (image_size - 1):
        raise ValueError("image_size must be a power of two and at least 16")


class Encoder(nn.Module):
    """Convolutional VAE encoder producing (mu, log_var) from RGB images."""

    def __init__(
        self,
        latent_dim: int = 128,
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

        final_channels = feature_maps * current_multiplier
        layers.extend(
            [
                nn.Conv2d(
                    final_channels,
                    final_channels,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(final_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

        self.body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(final_channels, latent_dim)
        self.log_var_head = nn.Linear(final_channels, latent_dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(images).flatten(start_dim=1)
        return self.mu_head(features), self.log_var_head(features)


class Decoder(nn.Module):
    """Convolutional VAE decoder producing RGB images in [-1, 1] from flat latents."""

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
        initial_multiplier = min(8, 2 ** (num_upsamples - 1))
        initial_channels = feature_maps * initial_multiplier

        self.project = nn.Linear(latent_dim, initial_channels)

        current_multiplier = initial_multiplier
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                initial_channels,
                initial_channels,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(initial_channels),
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).unsqueeze(-1).unsqueeze(-1)
        return self.network(x)


class VAE(nn.Module):
    """Variational Autoencoder with convolutional encoder and decoder."""

    def __init__(
        self,
        latent_dim: int = 128,
        encoder_features: int = 64,
        decoder_features: int = 64,
        image_size: int = 64,
        image_channels: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim, encoder_features, image_size, image_channels)
        self.decoder = Decoder(latent_dim, decoder_features, image_size, image_channels)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(images)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = (0.5 * log_var).exp()
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(images)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def initialize_vae_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
