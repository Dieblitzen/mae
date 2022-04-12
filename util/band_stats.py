import torch
import torchvision
import argparse
from tqdm import tqdm

from util.fmow_datasets import CustomDatasetFromImages, SentinelIndividualImageDataset


def get_mean_std(dataset):
    mean = torch.zeros(dataset[0][0].shape[1]).log()  # (c,)
    std = torch.zeros(dataset[0][0].shape[1]).log()  # (c,)
    count = 0
    for i in tqdm(range(len(dataset))):
        x, _ = dataset[i]  # (c, h, w)
        img_sum = x.sum((1, 2)).log()  # (c,)
        img_sum_sq = (x**2).sum((1, 2)).log()  # (c,)
        count += x[0].numel()

        mean = torch.logsumexp(torch.stack((mean, img_sum), dim=0), dim=0)  # (c,)
        std = torch.logsumexp(torch.stack((std, img_sum_sq), dim=0), dim=0)  # (c,)

    mean = torch.exp(mean)/count
    std = torch.sqrt(std/count - (mean ** 2))

    print(f"Mean: {mean.tolist()}")
    print(f"StdDev: {std.tolist()}")


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

