import torch
from torch import nn

from blocks import ResBlockBottle


class SmallAZeroNN(nn.Module):
    """
    Old NN architecture using the specified AlphaGo Zero architecture (with less residual blocks).
    Has a 3x3 convolution, then multiple ResNet blocks.
    Separates out into a policy (probabilities) network and a value network.
    """
    def __init__(self, height: int, width: int, is_direct: bool, num_players: int,
                 inner_channels: int = 256, restower_blocks: int = 5, **kwargs):
        super(SmallAZeroNN, self).__init__()
        if not is_direct:
            raise NotImplementedError

        tot_area = height * width
        num_actions = tot_area

        self.block1 = nn.Sequential(
            nn.Conv2d(num_players + 1, inner_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )

        resblock_list = []
        for _ in range(restower_blocks):
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
