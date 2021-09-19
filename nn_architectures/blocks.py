from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    """
    ResNet block for use in the main portion of the NN
    Two 3x3 convolutions followed by a residual connection
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlock, self).__init__()
        self.main_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        orig_x = x
        x = self.main_block(x)
        if self.downsample is not None:
            orig_x = self.downsample(orig_x)
        x += orig_x
        return F.relu(x)


class ResBlockBottle(nn.Module):
    """
    ResNet block (with bottleneck, inspired by MobileNet) in order to reduce computation time
    Two 3x3 depth-wise convolutions surrounded by 1x1 convolutions, and followed by a residual connection
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlockBottle, self).__init__()
        self.main_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            # Applies depth-wise 3x3 convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1, groups=in_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Another 3x3 depth-wise convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, groups=out_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        orig_x = x
        x = self.main_block(x)
        if self.downsample is not None:
            orig_x = self.downsample(orig_x)
        x += orig_x
        return F.relu(x)


class ResBlockBottleBias(nn.Module):
    """
    ResNet block (with bottleneck, inspired by MobileNet) in order to reduce computation time
    The difference between this class and the prior is that the conv layers w/o a batch-norm now have bias
    Two 3x3 depth-wise convolutions surrounded by 1x1 convolutions, and followed by a residual connection
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlockBottleBias, self).__init__()
        self.main_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            # Applies depth-wise 3x3 convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1, groups=in_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Another 3x3 depth-wise convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, groups=out_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        orig_x = x
        x = self.main_block(x)
        if self.downsample is not None:
            orig_x = self.downsample(orig_x)
        x += orig_x
        return F.relu(x)
