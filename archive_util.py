import numpy as np
import torch
from torch import nn
from torch.optim import optimizer

import os


def save_model(model: nn.Module, epoch: int, model_name: str, dir_path: str):
    file_name = model_name + '_' + str(epoch) + ".pth"
    torch.save(model.state_dict(), os.path.join(dir_path, file_name))


def save_optim(optim: optimizer, epoch: int, model_name: str, dir_path: str):
    file_name = model_name + '_optim_' + str(epoch) + ".pth"
    torch.save(optim.state_dict(), os.path.join(dir_path, file_name))


def load_model(model: nn.Module, epoch: int, model_name: str, dir_path: str):
    file_name = model_name + '_' + str(epoch) + ".pth"
    model.load_state_dict(torch.load(os.path.join(dir_path, file_name)))










