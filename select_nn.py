from torch import nn

from nn_architectures.conv_only import ConvSkipNN
from nn_architectures.fc_connection import FCConnectionsNN
from nn_architectures.small_azero import SmallAZeroNN


def get_nn(config: dict) -> nn.Module:
    if "nn_type" in config:
        if config["nn_type"] == "FCConnectionsNN":
            return FCConnectionsNN(**config)
        elif config["nn_type"] == "ConvSkipNN":
            return ConvSkipNN(**config)
        elif config["nn_type"] == "SmallAZeroNN":
            return SmallAZeroNN(**config)
        else:
            print("NN Type not available.")
            raise NotImplementedError
    else:  # Accommodating for older config files without "nn_type" key
        is_res_tower = config.get("restower", True)
        if is_res_tower:
            return SmallAZeroNN(**config)
        else:
            return ConvSkipNN(**config)
