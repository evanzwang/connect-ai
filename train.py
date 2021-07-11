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


def run_batch(batch: list[torch.Tensor, torch.Tensor, torch.Tensor], pvnn: nn.Module, optim: torch.optim.Optimizer):
    input_state = batch[0].to(device=device).float()
    r_prob = batch[1].to(device=device)
    r_val = batch[2].to(device=device)

    loss = compute_losses(pvnn(input_state), (r_prob, r_val))

    optim.zero_grad()
    loss.backward()
    optim.step()


def train(config: dict, dir_path: str):
    pvnn = ProbValNN(**config).to(device=device)
    pvnn.eval()
    bm = BoardManager(**config)

    mem_data = MemoryDataset(**config)
    dl = DataLoader(mem_data, shuffle=True, pin_memory=True,
                    num_workers=config["num_workers"], batch_size=config["batch_size"])
    optim = torch.optim.Adam(pvnn.parameters(), lr=config["learning_rate"], weight_decay=config["l2_reg"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.1)

    print("Done setup.")

    for epoch_num in range(1, config["num_epochs"] + 1):
        print(f"Epoch {epoch_num}")
        # Gets starting player, chooses from 0 -> num_players-1 (inclusive)
        curr_player = random.randrange(0, config["num_players"]) + 1
        curr_board = bm.blank_board()

        new_data = []

        print("Current move: ", end='')
        while 1:
            tree = MCST(pvnn, bm, device=device, **config)
            for _ in range(config["mcst_steps"]):
                tree.search(curr_board, curr_player)

            a_prob = tree.action_probs(curr_board, curr_player, 1)
            new_data.append([curr_board, a_prob, curr_player])
            print(f"{len(new_data)}")
            action = np.random.choice(a_prob.size, p=a_prob)

            curr_board, win_status = bm.take_action(curr_board, action, curr_player)
            curr_player = bm.next_player(curr_player)

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
            trained_samples = 0
            for batch in dl:
                run_batch(batch, pvnn, optim)
                trained_samples += batch[0].shape[0]
                if trained_samples >= config["max_samples_per_train"]:
                    break
            pvnn.eval()

            print(f"Playing random WR: {play_random(pvnn, device, **config)}")

        if epoch_num % config["epochs_per_save"] == 0:
            save_model(pvnn, epoch_num, config["model_name"], dir_path)
        if epoch_num % config["lr_decay_rate"]:
            scheduler.step()


def main(config_path: str):
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    print("Starting training.")
    train(config, os.path.dirname(config_path))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "experiments/first/first_config.yml"
    main(path)
