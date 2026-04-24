import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class CNNTransformer(nn.Module):
    """
    Hybrid CNN + Transformer model for speech command classification.

    Input:  (B, 1, 128, 112) log-Mel spectrogram
    CNN:    extracts local time-frequency features
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int = 12,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.cnn_features = nn.Sequential(
            ConvBlock(1, 32),      # (B, 32, 64, 56)
            ConvBlock(32, 64),     # (B, 64, 32, 28)
            ConvBlock(64, 128),    # (B, 128, 16, 14)
        )

        self.proj = nn.Conv2d(
            in_channels=128,
            out_channels=embed_dim,
            kernel_size=1,
        )

        num_tokens = 16 * 14 # W razie zmian w CNN, należy zaktualizować tę wartość

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, n_heads, mlp_ratio)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_features(x)
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])
        x = self.dropout(x)

        return self.head(x)


# if __name__ == "__main__":
#     model = CNNTransformer(num_classes=12)

#     x = torch.randn(4, 1, 128, 112)
#     y = model(x)

#     print("Input shape:", x.shape)
#     print("Output shape:", y.shape)