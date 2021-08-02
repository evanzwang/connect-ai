import numpy as np
import torch
from torch import nn
import yaml

import os

from archive_util import load_model
from nn import ProbValNN
from versus import play_model_human


def play_model(config: dict, config_path: str, epoch_num: int):
    """
    Function to play a select model, given the config and select epoch number
    :param config: Dictionary with all the config parameters
    :param config_path: Path to the config directory (used to access NN weights)
    :param epoch_num: Epoch number to select
    """
    pvnn = ProbValNN(**config).to(device=device)
    load_model(pvnn, epoch_num, config["model_name"], config_path)
    play_model_human(pvnn, device, **config)


def main(config_path: str):
    # Load config parameters
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    print("Starting play.")
    play_model(config, os.path.dirname(config_path), epoch)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Change the below for different models
    path = "experiments/ninth/nine.yml"
    epoch = 5500
    main(path)
