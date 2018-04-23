from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np

from dataset import data_loader
from dataset.data_loader import get_dataloader


def get_args():
    args = ArgumentParser()
    args.add_argument("dataset_name", choices=data_loader.dataset_list)
    args.add_argument("--data_aug", default="simple-no-norm")
    return args.parse_args()


def get_stats(input_loader):
    n_batches = len(input_loader)
    std = np.zeros((n_batches, 3))
    mean = np.zeros((n_batches, 3))

    for i, (data, _) in enumerate(tqdm(input_loader)):
        std[i] = data.numpy().std(axis=(0, 2, 3))
        mean[i] = data.numpy().mean(axis=(0, 2, 3))
    print("STD:", std.mean(axis=0))
    print("MEAN:", mean.mean(axis=0))


if __name__ == "__main__":
    args = get_args()
    input_loader = get_dataloader(args.dataset_name, 1000, 28, args.data_aug, None)
    get_stats(input_loader)
