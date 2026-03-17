import torch

from .mobilenetv2 import MobileNetV2
from .resnet20 import ResNet20
from .custom_cnn import CustomCNN

MODEL_REGISTRY: dict[str, type[torch.nn.Module]] = {
    "mobilenetv2": MobileNetV2,
    "resnet20": ResNet20,
    "custom_cnn": CustomCNN,
}