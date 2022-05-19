import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, download_url

import os
import pandas as pd
import numpy as np
import warnings
import random
from glob import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from PIL import Image
import rasterio
from rasterio import logging

log = logging.getLogger()
log.setLevel(logging.ERROR)

ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
SOME_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
RGB_BANDS = ['B04', 'B03', 'B02']

BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B11': 1594.42694882,
        'B12': 1009.32729131
    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B11': 1079.19066363,
        'B12': 818.86747235
    }
}

LABELS = [
    'Agro-forestry areas', 'Airports',
    'Annual crops associated with permanent crops', 'Bare rock',
    'Beaches, dunes, sands', 'Broad-leaved forest', 'Burnt areas',
    'Coastal lagoons', 'Complex cultivation patterns', 'Coniferous forest',
    'Construction sites', 'Continuous urban fabric',
    'Discontinuous urban fabric', 'Dump sites', 'Estuaries',
    'Fruit trees and berry plantations', 'Green urban areas',
    'Industrial or commercial units', 'Inland marshes', 'Intertidal flats',
    'Land principally occupied by agriculture, with significant areas of '
    'natural vegetation', 'Mineral extraction sites', 'Mixed forest',
    'Moors and heathland', 'Natural grassland', 'Non-irrigated arable land',
    'Olive groves', 'Pastures', 'Peatbogs', 'Permanently irrigated land',
    'Port areas', 'Rice fields', 'Road and rail networks and associated land',
    'Salines', 'Salt marshes', 'Sclerophyllous vegetation', 'Sea and ocean',
    'Sparsely vegetated areas', 'Sport and leisure facilities',
    'Transitional woodland/shrub', 'Vineyards', 'Water bodies', 'Water courses'
]

NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]

GROUP_LABELS = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters'
}


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class Bigearthnet(Dataset):
    url = 'http://bigearth.net/downloads/BigEarthNet-v1.0.tar.gz'
    subdir = 'BigEarthNet-v1.0'
    list_file = {
        'train': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-train.txt',
        'val': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-val.txt',
        'test': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-test.txt'
    }
    bad_patches = [
        'http://bigearth.net/static/documents/patches_with_seasonal_snow.csv',
        'http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv'
    ]

    def __init__(self, root, split, bands=None, transform=None, target_transform=None, download=False, use_new_labels=True):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else SOME_BANDS
        self.in_c = len(self.bands)
        def get_trans():
            t = []
            if split == 'train':
                t.append(transforms.ToTensor())
                t.append(transforms.RandomResizedCrop((96, 96), scale=(0.5, 1.0), interpolation=3))  # 3 is bicubic
                t.append(transforms.RandomHorizontalFlip())
                return transforms.Compose(t)
            else:
                t.append(transforms.ToTensor())
                t.append(
                    transforms.Resize(96, interpolation=3)  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop((96, 96)))
                return transforms.Compose(t)

        self.transform = transform if transform is not None else get_trans()
        self.target_transform = target_transform
        self.use_new_labels = use_new_labels

        if download:
            download_and_extract_archive(self.url, self.root)
            download_url(self.list_file[self.split], self.root, f'{self.split}.txt')
            for url in self.bad_patches:
                download_url(url, self.root)

        bad_patches = set()
        for url in self.bad_patches:
            filename = Path(url).name
            with open(self.root / filename) as f:
                bad_patches.update(f.read().splitlines())

        self.samples = []
        with open(self.root / f'{self.split}.txt') as f:
            for patch_id in f.read().splitlines():
                if patch_id not in bad_patches:
                    self.samples.append(self.root / self.subdir / patch_id)


    def __getitem__(self, index):
        path = self.samples[index]
        patch_id = path.name

        channels = []
        for b in self.bands:
            ch = rasterio.open(path / f'{patch_id}_{b}.tif').read(1)
            ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
            channels.append(ch)
        # img = np.dstack(channels)
        # img = Image.fromarray(img)

        with open(path / f'{patch_id}_labels_metadata.json', 'r') as f:
            labels = json.load(f)['labels']
        if self.use_new_labels:
            target = self.get_multihot_new(labels)
        else:
            target = self.get_multihot_old(labels)

        if self.transform is not None:
            img = [self.transform(ch) for ch in channels]
            # img = self.transform(img)
            img = torch.cat(img, dim=0)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # print(target.shape)

        return img, target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_multihot_old(labels):
        target = np.zeros((len(LABELS),), dtype=np.float32)
        for label in labels:
            target[LABELS.index(label)] = 1
        return target

    @staticmethod
    def get_multihot_new(labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
        for label in labels:
            if label in GROUP_LABELS:
                target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
            elif label not in set(NEW_LABELS):
                continue
            else:
                target[NEW_LABELS.index(label)] = 1
        return target


if __name__ == '__main__':
    import os
    import argparse
    from data import make_lmdb, random_subset

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/yzcong')
    parser.add_argument('--save_dir', type=str, default='test')
    args = parser.parse_args()

    train_dataset = Bigearthnet(
        root=args.data_dir,
        split='train'
    )
    train_dataset = random_subset(train_dataset, 0.1, 42)
    # make_lmdb(train_dataset, lmdb_file=os.path.join(args.save_dir, 'train.lmdb'))

    val_dataset = Bigearthnet(
        root=args.data_dir,
        split='val'
    )

    print(len(train_dataset))
    print(len(val_dataset))

    for samp in train_dataset:
        print(samp)
        print(np.array(samp[0]).shape, samp[1])
        break
    # make_lmdb(val_dataset, lmdb_file=os.path.join(args.save_dir, 'val.lmdb'))


# class SentinelIndividualImageDataset(SatelliteDataset):
#     '''fMoW Dataset'''
#     label_types = ['value', 'one-hot']
#     mean = [1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
#             1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
#             1972.62420416,  582.72633433,   14.77112979, 1732.16362238, 1247.91870117]
#     std = [633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
#            948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
#            1364.38688993,  472.37967789,   14.3114637 , 1310.36996126, 1087.6020813]

#     def __init__(self,
#                  split_path,
#                  transform,
#                  years=[*range(2000, 2021)],
#                  categories=None,
#                  label_type='value',
#                  resize=64,
#                  masked_bands=None,
#                  dropped_bands=None):
#         """ Initialize the dataset.

#         Args:
#             csv_path (string): Path to the csv file with annotations.
#             years (list, optional): List of years to take images from, None to not filter
#             categories (list, optional): List of categories to take images from, None to not filter
#             pre_transform (callable, optional): Optional transformation to be applied to individual images
#                 immediately after loading in. If None, each image in a time series is resized to
#                 to the dimensions of the first image. If specified, transformation should take images to
#                 the same dimensions so they can be stacked together
#             transform (callable, optional): Optional transform to be applied
#                 on tensor images.
#             label_type (string): 'values' for single regression label, 'one-hot' for one hot labels
#             resize: Size to load images as
#             masked_bands: List of indices corresponding to which bands to mask out
#         """
#         super().__init__(in_c=13)
#         self.df = pd.read_csv(csv_path) \
#             .sort_values(['category', 'location_id', 'timestamp'])

#         # Filter by category
#         self.categories = CATEGORIES
#         if categories is not None:
#             self.categories = categories
#             self.df = self.df.loc[categories]

#         # Filter by year
#         if years is not None:
#             self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
#             self.df = self.df[self.df['year'].isin(years)]

#         self.indices = self.df.index.unique().to_numpy()

#         self.transform = transform

#         if label_type not in self.label_types:
#             raise ValueError(
#                 f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:',
#                 ', '.join(self.label_types))
#         self.label_type = label_type

#         self.resize = resize

#         self.masked_bands = masked_bands
#         self.dropped_bands = dropped_bands
#         if self.dropped_bands is not None:
#             self.in_c = self.in_c - len(dropped_bands)

#     def __len__(self):
#         return len(self.df)

#     def open_image(self, img_path):
#         with rasterio.open(img_path) as data:
#             # img = data.read(
#             #     out_shape=(data.count, self.resize, self.resize),
#             #     resampling=Resampling.bilinear
#             # )
#             img = data.read()  # (c, h, w)

#         return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

#     def __getitem__(self, idx):
#         ''' Gets timeseries info for images in one area

#         Args:
#             idx: Index of loc in dataset

#         Returns dictionary containing:
#             'images': Images in tensor of dim Batch_SizexLengthxChannelxHeightxWidth
#                 Channel Order: R,G,B, NIR
#             'labels': Labels of each image. Depends on 'label_type'
#                 'regression': First year is -0.5, second year is 0.5, so on.
#                 'one-hot': Returns one-hot encoding of years
#                 'classification': for class labels by year where '0' is no construction (#labels = years)
#             'years': Years of each image in timeseries
#             'id': Id of image location
#             'type': Type of image location
#             'is_annotated': True if annotations for dates are provided
#             'year_built': Year built as labeled in dataset
#         '''
#         selection = self.df.iloc[idx]

#         # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
#         images = self.open_image(selection['image_path'])  # (h, w, c)
#         if self.masked_bands is not None:
#             images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

#         labels = self.categories.index(selection['category'])

#         img_as_tensor = self.transform(images)  # (c, h, w)
#         if self.dropped_bands is not None:
#             keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
#             img_as_tensor = img_as_tensor[keep_idxs, :, :]

#         sample = {
#             'images': images,
#             'labels': labels,
#             'image_ids': selection['image_id'],
#             'timestamps': selection['timestamp']
#         }
#         return img_as_tensor, labels