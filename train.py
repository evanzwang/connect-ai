import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

import os
import random

from archive_util import *
from dataset import MemoryDataset
from env import BoardManager
from mcst import MCST
from nn import ProbValNN, ProbValNNOld
from versus import play_baseline, play_random


def compute_losses(pred: tuple[torch.Tensor, torch.Tensor], actual: tuple[torch.Tensor, torch.Tensor],
                   val_weight: float = 0.04) -> torch.Tensor:
    """
    Computes MSE loss on the state values, and cross-entropy on the probabilities
    :param pred: The predicted NN values, dimension ([batch_size, num_actions], [batch_size, 1])
    :param actual: The target values, dimension ([batch_size, num_actions], [batch_size, 1])
    :param val_weight: How much to weight the value loss
    :return: The loss, as a 0-dimensional PyTorch tensor
    """
    ans = torch.square(actual[1] - pred[1]).reshape(-1) * val_weight - torch.mean(actual[0] * torch.log(pred[0]), dim=1)
    return ans.mean()


def run_batch(batch: list[torch.Tensor], pvnn: nn.Module, optim: torch.optim.Optimizer,
              val_weight: float = 0.04) -> float:
    """
    Trains model on a batch
    :param batch: The batch of data, as a list of board states, action probabilities, state values, and epoch number
    Dimension: [[batch_size, num_players+1, height, width], [batch_size, num_actions], [batch_size], [batch_size, 1]]
    :param pvnn: The NN object to be trained
    :param optim: The optimizer
    :param val_weight: How much to weight the value loss
    :return: The loss (as a float for recording purposes)
    """
    # Separating out the batch
    input_state = batch[0].to(device=device).float()
    r_prob = batch[1].to(device=device)
    r_val = batch[2].to(device=device).reshape(-1, 1)

    # Loss function to compare the NN predictions vs target values
    loss = compute_losses(pvnn(input_state), (r_prob, r_val), val_weight=val_weight)
    # Backpropagation
    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.item()


def train(config: dict, dir_path: str):
    """
    Trains the model
    :param config: The config parameters
    :param dir_path: The config directory (for saving models and record file)
    """
    # Initializing baseline NN
    versus_nn = ProbValNNOld(**versus_config).to(device=device)
    load_model(versus_nn, versus_epoch, versus_config["model_name"], os.path.dirname(versus_path))
    versus_nn.eval()

    record_path = os.path.join(dir_path, config["model_name"] + "_record.txt")
    until_train = config.get("samples_before_train", 0)

    # Initializing training NN
    pvnn = ProbValNN(**config).to(device=device)
    if pretraining_weights is not None:
        pvnn.load_state_dict(torch.load(pretraining_weights))
    pvnn.eval()

    # Other setup
    bm = BoardManager(**config)
    mem_data = MemoryDataset(**config)
    dl = DataLoader(mem_data, shuffle=True, pin_memory=True,
                    num_workers=config["num_workers"], batch_size=config["batch_size"])
    optim = torch.optim.Adam(pvnn.parameters(), lr=config["learning_rate"], weight_decay=config["l2_reg"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.65)

    value_weight = config.get("value_weight", 0.04)

    print("Done setup.")

    for epoch_num in range(1, config["num_epochs"] + 1):
        print(f"Epoch {epoch_num}")
        # Gets starting player, chooses from 1 -> num_players (inclusive)
        curr_player = random.randrange(0, config["num_players"]) + 1
        curr_board = bm.blank_board()
        new_data = []

        while 1:
            tree = MCST(pvnn, bm, device=device, **config)
            # Runs MC search for specified iterations
            for _ in range(config["mcst_steps"]):
                tree.search(curr_board, curr_player)

            # Takes an action based on the MCST action probabilities
            a_prob = tree.action_probs(curr_board, curr_player, 1)
            action = np.random.choice(a_prob.size, p=a_prob)

            # Records the board state, "target" action probabilities, and the player making the action for training
            new_data.append([curr_board, a_prob, curr_player])

            # Makes a move and moves to the next player
            curr_board, win_status = bm.take_action(curr_board, action, curr_player)
            curr_player = bm.next_player(curr_player)

            if len(new_data) % 10 == 0:
                print(len(new_data))

            if win_status:  # If the game ended
                reward = 0 if win_status == -2 else config["win_reward"]
                for el in new_data:
                    # Is -reward if the player wasn't the player that got the win
                    relative_reward = (-1) ** (el[2] != win_status) * reward
                    # Calculates all equivalences to augment data, and inserts them into the Dataset
                    all_equivs = bm.all_equivalences(el[0], el[1])
                    for equiv_state, equiv_prob in all_equivs:
                        mem_data.add(
                            (bm.onehot_perspective(equiv_state, el[2]),
                             equiv_prob,
                             relative_reward,
                             epoch_num)
                        )
                break

        if epoch_num % config["games_per_batch"] == 0 and len(mem_data) >= until_train:
            pvnn.train()
            trained_batches = 0
            l_val = 0
            # Trains network to predict "target" probabilities and state values
            for batch in dl:
                l_val += run_batch(batch, pvnn, optim, val_weight=value_weight)
                trained_batches += 1
                print(f"Batch {trained_batches} epochs: {batch[3][:10]}")
                # Runs only select number of training examples
                if trained_batches * config["batch_size"] >= config["max_samples_per_train"]:
                    break
            pvnn.eval()
            update_stats(record_path, f"Epoch {epoch_num} Loss: {l_val / trained_batches}")

        if epoch_num % config["epochs_per_save"] == 0:  # Saves model weights
            if len(mem_data) >= until_train:
                save_model(pvnn, epoch_num, config["model_name"], dir_path)

            fwr, swr, wr = play_baseline(pvnn, versus_nn, device, config, versus_config, num_trials=15)
            print(f"Playing random WR: {wr}")
            # Detailed WR in the record text file
            update_stats(record_path, f"Epoch {epoch_num} First Move WR: {fwr} || Second Move WR: {swr}")
            update_stats(record_path, f"Epoch {epoch_num} Base Overall WR: {wr}")

        if epoch_num % config["lr_decay_rate"] == 0:  # Updates LR
            scheduler.step()


def update_stats(file_path: str, write_string: str):
    """
    Appends a string to the record file (for future reference)
    Used for recording loss and win rates
    :param file_path: The path to write to
    :param write_string: The string that will be written
    """
    with open(file_path, "a+") as f:
        f.write(write_string + "\n")


def main(config_path: str):
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    print("Starting training.")
    train(config, os.path.dirname(config_path))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Config path
    path = "experiments/fixed/fixed.yml"
    # Set to a path with weights if model is building of previous weights
    pretraining_weights = None

    # Baseline model to measure off
    versus_path = "experiments/fifth_night/fifthredo.yml"
    versus_epoch = 3000
    with open(versus_path, "r") as y:
        versus_config = yaml.safe_load(y)

    main(path)
