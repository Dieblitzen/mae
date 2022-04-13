import concurrent.futures
import math
import torch
import torchvision
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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


def get_mean_std_parrallel(dataset):
    mean = torch.zeros(dataset[0][0].shape[0]).log()  # (c,)
    std = torch.zeros(dataset[0][0].shape[0]).log()  # (c,)
    count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        future_to_i = {ex.submit(get_sum_count, i, dataset) for i in range(len(dataset))}
        for future in concurrent.futures.as_completed(future_to_i):
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


def passed_args():
    parser = argparse.ArgumentParser(description="Get band stats")
    parser.add_argument('--dataset_type', type=str, default='rgb',
                        help='Sentinel or rgb')
    parser.add_argument('--dataset', type=str, required=True)
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
    mean, std = get_mean_std(dataset)
    print(f"Mean: {mean.tolist()}")
    print(f"StdDev: {std.tolist()}")
