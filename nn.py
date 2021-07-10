import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
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


class ProbValNN(nn.Module):
    def __init__(self, width: int, height: int, is_direct: bool, num_players: int,
                 inner_channels: int = 256, **kwargs):
        super(ProbValNN, self).__init__()
        # Probably want conv net i think, so it is like 19x19 with some convolutions
        # lol want to do resnet? lol could do u-net architecture thing
        # nah do convs then skip connection then FC layers, softmax
        # also have separate output for the probability / state value
        # output 19x19 grid
        #
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
        for _ in range(1):
            resblock_list.append(ResBlock(inner_channels, inner_channels))
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
