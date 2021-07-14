import numpy as np
import time
import random



def timetest():
    peen = {}

    cool_keys = []

    NUM_ITER = 10000
    for i in range(NUM_ITER):
        cool_keys.append((np.random.randn(19, 19) * 3).astype(np.uint8))
        peen[cool_keys[i].tobytes()] = random.randrange(0, 1000)

    n_arrays = []
    for i in range(NUM_ITER):
        n_arrays.append(np.random.randint(1000, size=391))


    NUM_TEST = 100000

    tic = time.perf_counter_ns()
    for i in range(NUM_TEST):
        rand_ind = random.randrange(0, NUM_ITER)
        bee = peen[cool_keys[rand_ind].tobytes()]
    toc = time.perf_counter_ns()
    print(toc-tic)

    tic = time.perf_counter_ns()
    for i in range(NUM_TEST):
        rand_ind = random.randrange(0, NUM_ITER)
        bee = n_arrays[rand_ind].sum()
    toc = time.perf_counter_ns()
    print(toc-tic)

def mem_testlist():
    max_iter = int(1e9)
    data = []
    for i in range(max_iter):
        sdata = (np.random.randint(2, size=(3, 19, 19), dtype=bool), np.random.rand(361), 1.)
        data.append(sdata)
        if i % 100000 == 0:
            print(i)


def mem_testarray():

    pass

if __name__ == "__main__":

    # mem_testlist()

    from dataset import MemoryDataset
    from torch.utils.data import DataLoader

    mem_data = MemoryDataset(max_memory=4, random_replacement=False)
    pee = DataLoader(mem_data, shuffle=True, pin_memory=True, batch_size=2, num_workers=8)
    mem_data.add((np.zeros((5, 5)), np.zeros(10), 1))
    mem_data.add((np.random.rand(5, 5), np.zeros(10), 1))
    print(len(pee), len(mem_data))

    for i, samp in enumerate(pee):
        print(samp)
        print(type(samp))

    mem_data.add((np.ones((5, 5)), np.zeros(10), 1))
    mem_data.add((np.random.randint(5, size=(5, 5)), np.zeros(10), 1))
    mem_data.add((np.arange(25).reshape(5, 5), np.zeros(10), 1))
    mem_data.add((np.ones((5, 5))*-1, np.zeros(10), 1))
    print(len(pee), len(mem_data))
    for i, samp in enumerate(pee):
        print(samp)
