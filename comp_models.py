import numpy as np
import torch
from torch import nn
import yaml

import math
import os

from archive_util import load_model
from select_nn import get_nn
from versus import play_baseline


def pit_models(config: dict, num_trials: int = 100):
    """
    Function to compare a selected model versus a baseline model, given the config of the selected NN.
    They play multiple rounds against each other, and the win rate is recorded.
    :param config: Dictionary with all the config parameters of the selected NN
    :param num_trials: Int specifying how many games to run
    """
    pvnn = get_nn(config).to(device=device)
    load_model(pvnn, epoch, config["model_name"], os.path.dirname(path))
    pvnn.eval()

    versus_nn = get_nn(versus_config).to(device=device)
    load_model(versus_nn, versus_epoch, versus_config["model_name"], os.path.dirname(versus_path))
    versus_nn.eval()

    fwr, swr, wr = play_baseline(pvnn, versus_nn, device, config, versus_config, num_trials)
    # Determining the percentage of first-move games (and thus how many there were)
    if math.isclose(fwr, swr):  # If the first-move WR is the same as second-move WR, you can't determine the percents
        t_f = "N/A"
        t_s = "N/A"
    else:
        percent_s = (wr - fwr) / (swr - fwr)  # Algebraic determination of the percentage of second-move games
        t_s = round(percent_s * num_trials)
        t_f = num_trials - t_s

    # Gives first move statistics as well as overall
    print(f"With current model first: {fwr} for {t_f} games.")
    print(f"With current model second: {swr} for {t_s} games.")
    print(f"All in all: {wr} for {num_trials} games.")


def main(config_path: str):
    # Load config parameters
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    print("Starting play.")
    pit_models(config, 100)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Change the below for different models
    path = "experiments/new_five/nfive.yml"
    epoch = 6000

    # Baseline model to measure off
    versus_path = "experiments/fifth_night/fifthredo.yml"
    versus_epoch = 3000
    with open(versus_path, "r") as y:
        versus_config = yaml.safe_load(y)

    main(path)


