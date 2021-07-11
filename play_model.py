import numpy as np
import torch
from torch import nn
import yaml

import os

from nn import ProbValNN
from versus import play_model_human
from archive_util import load_model


def play_model(config: dict, config_path: str, epoch_num: int):
    pvnn = ProbValNN(**config)
    load_model(pvnn, epoch_num, config["model_name"], config_path)
    play_model_human(pvnn, device, **config)


def main(config_path: str):
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    print("Starting training.")
    play_model(config, os.path.dirname(config_path), 240)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "blank_config.yml"
    main(path)


