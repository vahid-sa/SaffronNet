from __future__ import print_function, division
import sys
from os import path as osp
import torch
import numpy as np
import random
import csv
import json

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler


import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2 as cv

from .settings import NUM_VARIABLES


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, filenames_path, partition, class_list, images_dir, image_extension=".jpg", transform=None):
        """
        Args:
            filenames_path (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.filenames = filenames_path
        self.class_list = class_list
        self.transform = transform
        self.img_dir = images_dir
        self.ext = image_extension

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(
                    csv.reader(file, delimiter=','))
        except ValueError as e:
            raise(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, ctr_x, ctr_y, alpha, class_name
        try:
            with open(self.filenames, "r") as file:
                string_filenames = file.read()
            filenames = json.loads(string_filenames)
        except ValueError as e:
            raise(ValueError(
                'incorrect filenames path: {}: {}'.format(self.filenames, e)))
        assert partition in filenames.keys(), "incorrect partition"
        self.image_names = filenames[partition]

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise ValueError(fmt.format(e))

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError(
                    'line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = self._parse(
                class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError(
                    'line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        sample = {'img': img}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_path = osp.join(self.img_dir, self.image_names[image_index] + self.ext)
        img = cv.imread(image_path)

        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        return img.astype(np.float32)/255.0

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = cv.imread(self.image_names[image_index], cv.IMREAD_GRAYSCALE)
        height, width = image.shape
        return float(width) / float(height)


def collater(data):

    imgs = [s['img'] for s in data]
    scales = [s['scale'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    """Resizer: checked!
    """

    def __call__(self, sample, min_side=608, max_side=1024):
        image = sample['img']

        rows, cols, cns = image.shape

        # smallest_side = min(rows, cols)

        # # rescale the image so the smallest side is min_side
        # scale = min_side / smallest_side

        # # check if the largest side is now greater than max_side, which can happen
        # # when images have a large aspect ratio
        # largest_side = max(rows, cols)

        # if largest_side * scale > max_side:
        #     scale = max_side / largest_side
        scale = 1

        # resize the image with the computed scale
        image = skimage.transform.resize(
            image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros(
            (rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        return {'img': torch.from_numpy(new_image), 'scale': scale}


class Normalizer(object):
    """ Normalizer: checked!
    """

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image = sample['img']
        return {'img': ((image.astype(np.float32)-self.mean)/self.std)}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
