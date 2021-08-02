import numpy as np
import torch
from torch import nn
from torch.optim import optimizer

import os


def save_model(model: nn.Module, epoch: int, model_name: str, dir_path: str):
    """
    Saves a given model's weights. The file path it is saved to is determined by the other parameters
    :param model: The PyTorch model
    :param epoch: The current epoch of the model
    :param model_name: The model name
    :param dir_path: The base directory the model will be saved under
    """
    file_name = model_name + '_' + str(epoch) + ".pth"
    torch.save(model.state_dict(), os.path.join(dir_path, file_name))


def save_optim(optim: optimizer, epoch: int, model_name: str, dir_path: str):
    """
    Saves a given optimizer's state. The file path it is saved to is determined by the other parameters
    :param optim: The PyTorch optimizer
    :param epoch: The current epoch of the optimizer
    :param model_name: The model name
    :param dir_path: The base directory the optimizer weights will be saved under
    """
    file_name = model_name + '_optim_' + str(epoch) + ".pth"
    torch.save(optim.state_dict(), os.path.join(dir_path, file_name))


def load_model(model: nn.Module, epoch: int, model_name: str, dir_path: str):
    """
    Loads a given model's weights (given the epoch, model name, and directory) into the nn.Module object
    :param model: The nn.Module object
    :param epoch: The epoch number of the saved weights
    :param model_name: The model name
    :param dir_path: The directory of the weights
    """
    file_name = model_name + '_' + str(epoch) + ".pth"
    model.load_state_dict(torch.load(os.path.join(dir_path, file_name)))










