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


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()

        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 256 -> 128 -> 10 (Last one is the number of classes)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        print("after block1:", x.shape)
        x = self.block2(x)
        print("after block2:", x.shape)
        x = self.block3(x)
        print("after block3:", x.shape)
        x = self.block4(x)
        print("after block4:", x.shape)

        x = self.pool(x)
        print("after pool:", x.shape)
        x = torch.flatten(x, 1)
        print("after flatten:", x.shape)
        x = self.classifier(x)

        return x
    
    
if __name__ == "__main__":
    model = CustomCNN()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(out.shape)