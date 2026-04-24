import torch.nn as nn

from project2_speechcommands.models.cnn_baseline import CNNBaseline
from project2_speechcommands.models.transformer import SpectrogramTransformer
from project2_speechcommands.models.cnn_transformer import CNNTransformer

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "cnn_baseline": CNNBaseline,
    "transformer": SpectrogramTransformer,
    "cnn_transformer": CNNTransformer,
}

__all__ = ["MODEL_REGISTRY", "CNNBaseline", "SpectrogramTransformer", "CNNTransformer"]
