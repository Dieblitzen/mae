import math
import torch
import torchvision
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from fmow_datasets import CustomDatasetFromImages, SentinelIndividualImageDataset


def get_mean_std(dataset):
    mean = torch.zeros(dataset[0][0].shape[0]).log()  # (c,)
    std = torch.zeros(dataset[0][0].shape[0]).log()  # (c,)
    count = 0
    for i in tqdm(range(len(dataset))):
        x, _ = dataset[i]  # (c, h, w)
        img_sum = x.sum((1, 2)).log()  # (c,)
        img_sum_sq = (x**2).sum((1, 2)).log()  # (c,)
        count += x[0].numel()

        mean = torch.logsumexp(torch.stack((mean, img_sum), dim=0), dim=0)  # (c,)
        std = torch.logsumexp(torch.stack((std, img_sum_sq), dim=0), dim=0)  # (c,)

    mean = torch.exp(mean - math.log(count))
    std = torch.sqrt(torch.exp(std - math.log(count)) - (mean ** 2))

    return mean, std


def get_sum_count(i, dataset):
    x, _ = dataset[i]  # (c, h, w)
    img_sum = x.sum((1, 2)).log()  # (c,)
    img_sum_sq = (x ** 2).sum((1, 2)).log()  # (c,)
    img_count = x[0].numel()
    return img_sum, img_sum_sq, img_count


def get_mean_std_parallel(dataset, num_workers):
    mean = torch.zeros(dataset[0][0].shape[0]).log()  # (c,)
    std = torch.zeros(dataset[0][0].shape[0]).log()  # (c,)
    count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        future_to_i = {ex.submit(get_sum_count, i, dataset): i for i in range(len(dataset))}
        for future in tqdm(as_completed(future_to_i)):
            try:
                img_sum, img_sum_sq, img_count = future.result()
            except Exception as e:
                print(f"Error on img {future_to_i[future]}")
                raise e
            else:
                mean = torch.logsumexp(torch.stack((mean, img_sum), dim=0), dim=0)  # (c,)
                std = torch.logsumexp(torch.stack((std, img_sum_sq), dim=0), dim=0)  # (c,)
                count += img_count

    mean = torch.exp(mean - math.log(count))
    std = torch.sqrt(torch.exp(std - math.log(count)) - (mean ** 2))

    return mean, std


def pixel_counts(dataset, sample=None):
    seq = tqdm(sample) if sample is not None else tqdm(range(len(dataset)))
    channel_counters = defaultdict(Counter)
    for i in seq:
        x, _ = dataset[i]  # (c, h, w)

        for j, xc in enumerate(x):
            channel_counters[j].update(xc.reshape(-1))
    return channel_counters


def get_pixel_count_per_channel(i, dataset):
    x, _ = dataset[i]  # (c, h, w)
    channel_counters = defaultdict(Counter)
    for j, xc in enumerate(x):
        channel_counters[j].update(xc.reshape(-1))
    return channel_counters


def pixel_counts_parallel(dataset, num_workers, sample=None):
    channel_counters = defaultdict(Counter)

    seq = sample if sample is not None else range(len(dataset))
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        future_to_i = {ex.submit(get_pixel_count_per_channel, i, dataset): i for i in seq}
        for future in tqdm(as_completed(future_to_i)):
            try:
                img_channel_counters = future.result()
            except Exception as e:
                print(f"Error on img {future_to_i[future]}")
                raise e
            else:
                for c, counter in img_channel_counters.items():
                    channel_counters[c].update(counter)
    return channel_counters


def passed_args():
    parser = argparse.ArgumentParser(description="Get band stats")
    parser.add_argument('--dataset_type', type=str, default='rgb',
                        help='Sentinel or rgb')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--stat_type', type=str, default='mean_std',
                        choices=['mean_std', 'pixel_count'])
    return parser.parse_args()


if __name__ == "__main__":
    args = passed_args()
    if args.dataset_type == 'rgb':
        dataset_type = CustomDatasetFromImages
    elif args.dataset_type == 'sentinel':
        dataset_type = SentinelIndividualImageDataset
    else:
        raise NotImplementedError("No")

    transform = torchvision.transforms.ToTensor()
    dataset = dataset_type(args.dataset, transform)

    if args.stat_type == 'mean_std':
        # mean, std = get_mean_std(dataset)
        mean, std = get_mean_std_parallel(dataset, args.num_workers)
        print(f"Mean: {mean.tolist()}")
        print(f"StdDev: {std.tolist()}")
    elif args.stat_type == 'pixel_count':
        sample = np.random.choice(len(dataset), size=2000, replace=False)
        channel_pixel_counts = pixel_counts_parallel(dataset, args.num_workers, sample=sample)
        channel_pixel_counts = {c: dict(counter) for c, counter in channel_pixel_counts.items()}
        with open('pixel_counts.pkl', 'wb') as f:
            pickle.dump(channel_pixel_counts, f)
