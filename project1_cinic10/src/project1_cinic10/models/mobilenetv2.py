from torch import nn


class InvertedResidual(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int, expansion: int):
        super().__init__()

        self.use_residual = stride == 1 and in_channels == out_channels

        expanded_channels = in_channels * expansion

        self.expand = None
        if expansion != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(inplace=True)
            )

        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.expand(x) if self.expand is not None else x
        out = self.depthwise(out)
        out = self.pointwise(out)
        if self.use_residual:
            out = out + x
        return out


class MobileNetV2(nn.Module):

    def __init__(self, num_classes: int = 10, dropout: float = 0):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),  # Changed stride from 2 -> 1 to adjust for 32x32 input image
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.in_channels = 32

        bottleneck_config = (
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        )

        body_modules = []
        for t, c, n, s in bottleneck_config:
            body_modules.extend(self._make_stage(t, c, n, s))

        self.body = nn.Sequential(
            *body_modules
        )

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout),
            nn.Conv2d(1280, num_classes, kernel_size=1),
        )

    def _make_stage(self, t: int, c: int, n: int, s: int) -> list[nn.Module]:
        """
        :param t: expansion
        :param c: output channels
        :param n: amount of layers
        :param s: stride
        :return: bottleneck sequence made up of :py:class:`InvertedResidual`
        """
        blocks = [InvertedResidual(self.in_channels, c, s, t)]  # first block: transition
        self.in_channels = c  # update for subsequent blocks
        for _ in range(n - 1):
            blocks.append(InvertedResidual(c, c, 1, t))  # rest: same channels, stride 1, that means residual connections
        return blocks


    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        return x.flatten(1)