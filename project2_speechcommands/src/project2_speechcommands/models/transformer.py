import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Split (B, 1, 128, 112) spectrogram into non-overlapping patches via Conv2d.
    kernel_size=patch_size, stride=patch_size → (B, embed_dim, n_freq, n_time)
    Flatten spatial dims → (B, num_patches, embed_dim)

    With patch_size=(16,16): n_freq=8, n_time=7 → num_patches=56
    """

    def __init__(self, patch_size: tuple[int, int], embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, 8, 7)
        return x.flatten(2).transpose(1, 2)  # (B, 56, embed_dim)


class TransformerEncoderBlock(nn.Module):
    """
    Pre-norm Transformer encoder block:
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

    FFN: Linear → GELU → Linear
    Uses nn.MultiheadAttention(batch_first=True)
    """

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm based on AST architecture
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x


class SpectrogramTransformer(nn.Module):
    """
    AST-style Transformer on log-Mel spectrogram patches.

    Input:  (B, 1, 128, 112)
    Patches: 8×7 = 56 patches of 16×16
    Sequence: [CLS] + 56 patches = 57 tokens
    Output: (B, num_classes) — classify from CLS token
    """

    def __init__(
        self,
        num_classes: int = 12,
        patch_size: tuple[int, int] = (16, 16),
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        num_patches = (128 // patch_size[0]) * (112 // patch_size[1])
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, n_heads, mlp_ratio)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # Patch + linear projection
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)  # Prepend CLS token
        x = x + self.pos_embed  # Add positional encoding
        for block in self.blocks:  # N × (Norm → Attn → Add, Norm → FFN → Add)
            x = block(x)
        x = self.norm(x[:, 0])  # Extract + norm CLS token
        return self.head(x)  # Classify
