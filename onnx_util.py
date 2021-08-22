import torch
import yaml

import os

from archive_util import load_model
from nn import ProbValNN


def export_to_onnx(config: dict, config_path: str, epoch_num: int):
    """
    Function to load a select model to ONNX, given the config and select epoch number
    :param config: Dictionary with all the config parameters
    :param config_path: Path to the config directory (used to access NN weights)
    :param epoch_num: Epoch number to select
    """
    dummy_input = torch.randn((1, config["num_players"] + 1, config["height"], config["width"]),
                              dtype=torch.float, device=device)
    pvnn = ProbValNN(**config).to(device=device)
    load_model(pvnn, epoch_num, config["model_name"], config_path)
    pvnn.eval()

    file_name = config["model_name"] + '_' + str(epoch) + ".onnx"
    torch.onnx.export(pvnn, dummy_input, os.path.join(config_path, file_name))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Change the below for different models
    path = "experiments/new_five/nfive.yml"
    epoch = 6000
    with open(path, "r") as yml:
        config_dict = yaml.safe_load(yml)

    export_to_onnx(config_dict, os.path.dirname(path), epoch)

    print("Done.")
