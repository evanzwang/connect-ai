import torch
from torch import nn
import torch.nn.functional as F


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


class ProbValNN(nn.Module):
    """
    Updated NN architecture, using 5x5 kernels in the beginning, followed by multiple ResNet blocks.
    Also ignores the one-hot encoded first channel, which marks all the empty spaces.
    Concatenates with the original board at the end and applies another 5x5 filter to the concatenated features.
    Separates out into a policy (probabilities) network and a value network, as before.
    The policy network no longer has any FC layers.
    """
    def __init__(self, width: int, height: int, is_direct: bool, num_players: int,
                 inner_channels: int = 256, **kwargs):
        super(ProbValNN, self).__init__()

        if not is_direct:
            raise NotImplementedError

        tot_area = width * height
        num_actions = tot_area
        # First 5x5 convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(num_players, inner_channels, kernel_size=(5, 5), padding=2, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        # ResNet block tower
        resblock_list = []
        for _ in range(4):
            resblock_list.append(ResBlockBottleBias(inner_channels, inner_channels))
        self.res_tower = nn.Sequential(
            *resblock_list,
            nn.Conv2d(inner_channels, inner_channels // 4, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True)
        )
        # Layer used to run a convolution over the concatenated features
        self.concat = nn.Sequential(
            nn.Conv2d(inner_channels//4 + num_players, inner_channels//4, kernel_size=(5, 5), padding=2, bias=False),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(inner_channels // 4, inner_channels // 8, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(inner_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels // 8, 1, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(inner_channels // 4, 1, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(tot_area, num_actions),
            nn.ReLU(inplace=True),
            nn.Linear(num_actions, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Discards channel 0, which holds information about whether a space is empty
        concat_features = self.concat(torch.cat([self.res_tower(self.block1(x[:, 1:])), x[:, 1:]], dim=1))
        return self.policy_head(concat_features), self.value_head(concat_features)


class ProbValNNOld(nn.Module):
    """
    Old NN architecture. Still used to measure models against baseline (the baseline uses this architecture).
    Has a 3x3 convolution, then multiple ResNet blocks.
    Separates out into a policy (probabilities) network and a value network.
    """
    def __init__(self, width: int, height: int, is_direct: bool, num_players: int,
                 inner_channels: int = 256, **kwargs):
        super(ProbValNNOld, self).__init__()
        if not is_direct:
            raise NotImplementedError

        tot_area = width * height
        num_actions = tot_area

        self.block1 = nn.Sequential(
            nn.Conv2d(num_players + 1, inner_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )

        resblock_list = []
        for _ in range(5):
            resblock_list.append(ResBlockBottle(inner_channels, inner_channels))
        self.res_tower = nn.Sequential(*resblock_list)

        self.policy_head = nn.Sequential(
            nn.Conv2d(inner_channels, 2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(tot_area * 2, num_actions),
            nn.Softmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(inner_channels, 1, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(tot_area, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.res_tower(self.block1(x))
        return self.policy_head(features), self.value_head(features)
