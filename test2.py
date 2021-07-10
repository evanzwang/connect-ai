import random

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np



from dataset import MemoryDataset

from nn import ProbValNN
from env import BoardManager


def compute_losses(pred: tuple[torch.Tensor, torch.Tensor], actual: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    ans = torch.square(actual[1] - pred[1]) - torch.sum(actual[0] * torch.log(pred[0]), dim=1)
    return ans.mean()


def run_batch(batch: list[torch.Tensor, torch.Tensor, torch.Tensor], pnn: nn.Module, opt: torch.optim.Optimizer):
    input_state = batch[0].to(device=device).float()
    print(input_state.shape)
    r_prob = batch[1].to(device=device)
    r_val = batch[2].to(device=device)

    loss = compute_losses(pnn(input_state), (r_prob, r_val))

    opt.zero_grad()
    loss.backward()
    opt.step()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mem_data = MemoryDataset(1000000, False)
    dl = DataLoader(mem_data, shuffle=True, pin_memory=True, num_workers=8, batch_size=512)

    pvnn = ProbValNN(19, 19, True, 2).to(device=device)
    bm = BoardManager(19, 19, 5, True, 2)

    optim = torch.optim.Adam(pvnn.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.1)


    # pee_data = (3 * torch.rand((19, 19))).to(torch.uint8)
    # print(pee_data)
    # loloh = torch.from_numpy(bm.onehot_perspective(pee_data, 1))
    # loloh = loloh.unsqueeze(0).float()
    # print(loloh.shape)
    #
    # print(bruh(loloh))

    for _ in range(2048):
        b_state = (np.random.rand(19, 19) * 3).astype(np.uint8)

        prob = np.random.rand(19 * 19)
        prob /= prob.sum()
        v = (-1) ** (random.randrange(0, 1))
        mem_data.add((bm.onehot_perspective(b_state, 1), prob, v))

    print(mem_data.__len__())

    # pvnn.train()
    for be in dl:
        run_batch(be, pvnn, optim)
        break

    print("hi")

