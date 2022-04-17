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

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
              "aquaculture", "archaeological_site", "barn", "border_checkpoint",
              "burial_site", "car_dealership", "construction_site", "crop_field",
              "dam", "debris_or_rubble", "educational_institution", "electric_substation",
              "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
              "gas_station", "golf_course", "ground_transportation_station", "helipad",
              "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
              "lighthouse", "military_facility", "multi-unit_residential",
              "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
              "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility",
              "road_bridge", "runway", "shipyard", "shopping_mall",
              "single-unit_residential", "smokestack", "solar_farm", "space_facility",
              "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
              "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
              "wind_farm", "zoo"]


class CustomDatasetFromImages(Dataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


class CustomDatasetFromImagesTemporalClassification(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])  # [:16]
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])  # [:16]
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label, single_image_name.split('/')[-2])

    def __len__(self):
        return self.data_len


class CustomDatasetFromImagesTemporal(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]

        splt = single_image_name_1.rsplit('/', 1)
        base_path = splt[0]
        fname = splt[1]
        suffix = fname[-15:]
        prefix = fname[:-15].rsplit('_', 1)
        regexp = '{}/{}_*{}'.format(base_path, prefix[0], suffix)
        temporal_files = glob(regexp)
        temporal_files.remove(single_image_name_1)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
        else:
            single_image_name_2 = random.choice(temporal_files)

        img_as_img_1 = Image.open(single_image_name_1)
        img_as_tensor_1 = self.transforms(img_as_img_1)

        img_as_img_2 = Image.open(single_image_name_2)
        img_as_tensor_2 = self.transforms(img_as_img_2)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return ([img_as_tensor_1, img_as_tensor_2], single_image_label)

    def __len__(self):
        return self.data_len


class FMoWTemporalStacked(Dataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871,
            0.4182007312774658, 0.4214799106121063, 0.3991275727748871,
            0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697,
           0.28774282336235046, 0.27541765570640564, 0.2764017581939697,
           0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

        self.min_year = 2002

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]

        splt = single_image_name_1.rsplit('/', 1)
        base_path = splt[0]
        fname = splt[1]
        suffix = fname[-15:]
        prefix = fname[:-15].rsplit('_', 1)
        regexp = '{}/{}_*{}'.format(base_path, prefix[0], suffix)
        temporal_files = glob(regexp)
        temporal_files.remove(single_image_name_1)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]
        else:
            single_image_name_2 = random.choice(temporal_files)
            while True:
                single_image_name_3 = random.choice(temporal_files)
                if single_image_name_3 != single_image_name_2:
                    break

        img_as_img_1 = Image.open(single_image_name_1)
        img_as_tensor_1 = self.transforms(img_as_img_1)  # (3, h, w)

        img_as_img_2 = Image.open(single_image_name_2)
        img_as_tensor_2 = self.transforms(img_as_img_2)  # (3, h, w)

        img_as_img_3 = Image.open(single_image_name_3)
        img_as_tensor_3 = self.transforms(img_as_img_3)  # (3, h, w)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        img = torch.cat((img_as_tensor_1, img_as_tensor_2, img_as_tensor_3), dim=0)  # (9, h, w)
        return (img, single_image_label)

    def __len__(self):
        return self.data_len


class SentinelIndividualImageDataset(Dataset):
    '''fMoW Dataset'''
    label_types = ['value', 'one-hot']
    mean = [5.373300075531006, 4.644637107849121, 4.395181179046631, 4.455922603607178,
            4.955841064453125, 6.452561378479004, 7.242629051208496, 6.91213846206665,
            7.735781192779541, 2.28520131111145, 0.0579259991645813, 6.7927985191345215, 4.893798828125]
    std = [2.482947826385498, 2.5501344203948975, 2.792647361755371, 3.7852203845977783,
           3.7214980125427246, 4.345358848571777, 4.934760570526123, 4.835879325866699,
           5.350536823272705, 1.8524693250656128, 0.056123387068510056, 5.138705730438232, 4.265106201171875]

    def __init__(self,
                 csv_path,
                 transform,
                 years=[*range(2000, 2021)],
                 categories=None,
                 label_type='value',
                 resize=64):
        """ Initialize the dataset.

        Args:
            csv_path (string): Path to the csv file with annotations.
            years (list, optional): List of years to take images from, None to not filter
            categories (list, optional): List of categories to take images from, None to not filter
            pre_transform (callable, optional): Optional transformation to be applied to individual images
                immediately after loading in. If None, each image in a time series is resized to
                to the dimensions of the first image. If specified, transformation should take images to
                the same dimensions so they can be stacked together
            transform (callable, optional): Optional transform to be applied
                on tensor images.
            label_type (string): 'values' for single regression label, 'one-hot' for one hot labels
            resize: Size to load images as
        """
        self.df = pd.read_csv(csv_path) \
            .sort_values(['category', 'location_id', 'timestamp'])

        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(
                f'FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:',
                ', '.join(self.label_types))
        self.label_type = label_type

        self.resize = resize

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        ''' Gets timeseries info for images in one area

        Args:
            idx: Index of loc in dataset

        Returns dictionary containing:
            'images': Images in tensor of dim Batch_SizexLengthxChannelxHeightxWidth
                Channel Order: R,G,B, NIR
            'labels': Labels of each image. Depends on 'label_type'
                'regression': First year is -0.5, second year is 0.5, so on.
                'one-hot': Returns one-hot encoding of years
                'classification': for class labels by year where '0' is no construction (#labels = years)
            'years': Years of each image in timeseries
            'id': Id of image location
            'type': Type of image location
            'is_annotated': True if annotations for dates are provided
            'year_built': Year built as labeled in dataset
        '''
        selection = self.df.iloc[idx]

        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = self.open_image(selection['image_path']) / 255

        labels = self.categories.index(selection['category'])

        img_as_tensor = self.transform(images)

        sample = {
            'images': images,
            'labels': labels,
            'image_ids': selection['image_id'],
            'timestamps': selection['timestamp']
        }
        return img_as_tensor, labels


def build_fmow_dataset(is_train, args):
    csv_path = os.path.join(args.train_path if is_train else args.test_path)

    if args.dataset_type == 'rgb':
        mean = CustomDatasetFromImages.mean
        std = CustomDatasetFromImages.std
        transform = build_transform(is_train, args.input_size, mean, std)
        dataset = CustomDatasetFromImages(csv_path, transform)
    elif args.dataset_type == 'sentinel':
        mean = SentinelIndividualImageDataset.mean
        std = SentinelIndividualImageDataset.std
        transform = build_transform(is_train, args.input_size, mean, std)
        dataset = SentinelIndividualImageDataset(csv_path, transform)
    elif args.dataset_type == 'rgb_temporal_stacked':
        mean = FMoWTemporalStacked.mean
        std = FMoWTemporalStacked.std
        transform = build_transform(is_train, args.input_size, mean, std)
        dataset = FMoWTemporalStacked(csv_path, transform)
    elif args.dataset_type == 'combined':
        raise NotImplementedError("combined not yet implemented")
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
    print(dataset)

    return dataset


def build_transform(is_train, input_size, mean, std):
    # mean = IMAGENET_DEFAULT_MEAN
    # std = IMAGENET_DEFAULT_STD
    # train transform

    t = []
    if is_train:
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        )
        t.append(transforms.RandomHorizontalFlip())
        return transforms.Compose(t)

    # eval transform
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    t.append(
        transforms.Resize(size, interpolation=Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    # t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
