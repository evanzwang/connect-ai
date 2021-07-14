import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

import random

from archive_util import *
from dataset import MemoryDataset
from env import BoardManager
from mcst import MCST
from nn import ProbValNN
from versus import play_random


def compute_losses(pred: tuple[torch.Tensor, torch.Tensor], actual: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    ans = torch.square(actual[1] - pred[1]) - torch.sum(actual[0] * torch.log(pred[0]), dim=1)
    return ans.mean()


def run_batch(batch: list[torch.Tensor, torch.Tensor, torch.Tensor], pvnn: nn.Module, optim: torch.optim.Optimizer) \
        -> float:
    input_state = batch[0].to(device=device).float()
    r_prob = batch[1].to(device=device)
    r_val = batch[2].to(device=device)

    loss = compute_losses(pvnn(input_state), (r_prob, r_val))

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.item()


def train(config: dict, dir_path: str):
    record_path = os.path.join(dir_path, config["model_name"] + "_record.txt")

    pvnn = ProbValNN(**config).to(device=device)
    if pretraining_weights is not None:
        pvnn.load_state_dict(torch.load(pretraining_weights))
    pvnn.eval()
    bm = BoardManager(**config)

    mem_data = MemoryDataset(**config)
    dl = DataLoader(mem_data, shuffle=True, pin_memory=True,
                    num_workers=config["num_workers"], batch_size=config["batch_size"])
    optim = torch.optim.Adam(pvnn.parameters(), lr=config["learning_rate"], weight_decay=config["l2_reg"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.45)

    print("Done setup.")

    for epoch_num in range(1, config["num_epochs"] + 1):
        print(f"Epoch {epoch_num}")
        # Gets starting player, chooses from 0 -> num_players-1 (inclusive)
        curr_player = random.randrange(0, config["num_players"]) + 1
        curr_board = bm.blank_board()

        new_data = []

        while 1:
            tree = MCST(pvnn, bm, device=device, **config)
            for _ in range(config["mcst_steps"]):
                tree.search(curr_board, curr_player)

            a_prob = tree.action_probs(curr_board, curr_player, 1)
            new_data.append([curr_board, a_prob, curr_player])
            action = np.random.choice(a_prob.size, p=a_prob)

            curr_board, win_status = bm.take_action(curr_board, action, curr_player)
            curr_player = bm.next_player(curr_player)

            if len(new_data) % 10 == 0:
                print(len(new_data))

            if win_status:
                reward = 0 if win_status == -2 else config["win_reward"]
                for el in new_data:
                    mem_data.add(
                        (bm.onehot_perspective(el[0], el[2]),
                         el[1],
                         (-1) ** (el[2] != win_status) * reward)
                    )
                break

        if epoch_num % config["games_per_batch"] == 0 or epoch_num == 1:
            pvnn.train()
            trained_batches = 0
            l_val = 0
            for batch in dl:
                l_val += run_batch(batch, pvnn, optim)
                trained_batches += 1
                if trained_batches * config["batch_size"] >= config["max_samples_per_train"]:
                    break
            pvnn.eval()
            update_stats(record_path, f"Epoch {epoch_num} Loss: {l_val / trained_batches}")

        if epoch_num % config["epochs_per_save"] == 0:
            save_model(pvnn, epoch_num, config["model_name"], dir_path)
            print(f"Playing random WR: {play_random(pvnn, device, **config)}")
            update_stats(record_path, f"Epoch {epoch_num} Random WR: {play_random(pvnn, device, **config)}")

        if epoch_num % config["lr_decay_rate"] == 0:
            scheduler.step()


def update_stats(file_path: str, write_string: str):
    with open(file_path, "a+") as f:
        f.write(write_string + "\n")


def main(config_path: str):
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    print("Starting training.")
    train(config, os.path.dirname(config_path))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "experiments/fifth_night/fifthn.yml.yml"
    pretraining_weights = "experiments/fourth_night/fourthn_2500.yml"
    main(path)
