import torch
from torch import nn

from nn_architectures.blocks import ResBlockBottleBias


class FCConnectionsNN(nn.Module):
    """
    Fully-connected neural network with long skip connections.
    Merges the theory of SmallAZeroNN and ConvSkipNN.
    Uses 5x5 kernels in the beginning, followed by multiple ResNet blocks.
    Also ignores the one-hot encoded first channel, which marks all the empty spaces.
    Concatenates with the original board at the end and applies another 5x5 filter to the concatenated features.
    Separates out into a policy (probabilities) network and a value network, similar to AlphaZero.
    """
    def __init__(self, height: int, width: int, is_direct: bool, num_players: int,
                 inner_channels: int = 256, restower_blocks: int = 5, **kwargs):
        super(FCConnectionsNN, self).__init__()

        tot_area = height * width

        if not is_direct:
            num_actions = width
        else:
            num_actions = tot_area

        # First 5x5 convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(num_players, inner_channels, kernel_size=(5, 5), padding=2, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        # ResNet block tower
        resblock_list = []
        for _ in range(restower_blocks):
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
            nn.Conv2d(inner_channels // 4, 2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(tot_area * 2, num_actions),
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
