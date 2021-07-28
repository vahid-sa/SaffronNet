from __future__ import print_function, division
import sys
from os import path as osp
import os
import torch
import numpy as np
import random
import csv
import json

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torchvision

import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2 as cv

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

from utils.visutils import get_alpha, get_dots, DrawMode, std_draw_line
from retinanet.settings import NUM_VARIABLES



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
        self.transform = torchvision.transforms.Compose([Normalizer(), Resizer()])
        self.augment_transform = torchvision.transforms.Compose([Normalizer(), Augmenter(), Resizer()])
        self.augment = torchvision.transforms.Compose([Augmenter()])
        self.img_dir = images_dir
        self.ext = image_extension
        self.aug = Augmenter()

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

        img, img_name = self.load_image(idx)
        floated_img = img.astype(np.float32) / 255.0
        augmented_img = floated_img.copy()

        orig_sample = self.transform({'img': floated_img, 'name': img_name})
        aug_sample = self.augment_transform({'img': augmented_img, 'name': img_name})
        only_aug_sample = self.augment({'img': img, 'name': img_name})
        sample = orig_sample
        sample["aug_img"] = aug_sample["img"]
        sample["only_aug_img"] = only_aug_sample["img"]

        return sample

    def load_image(self, image_index):
        img_name = self.image_names[image_index]
        image_path = osp.join(self.img_dir, img_name + self.ext)
        img = cv.imread(image_path)

        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        return img, img_name

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
    names=[s["name"] for s in data]

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

    return {'img': padded_imgs, 'scale': scales, 'name': names}


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
        return {'img': torch.from_numpy(new_image), 'scale': scale, "name": sample["name"]}


class Normalizer(object):
    """ Normalizer: checked!
    """

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image = sample['img']
        return {'img': ((image.astype(np.float32)-self.mean)/self.std), "name": sample["name"]}


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


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    """ #
    """

    def __init__(self) -> None:
        super().__init__()
        ia.seed(3)
        self.seq = iaa.Sequential([
            # iaa.Affine(
            #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            #     rotate=(-15, 15),
            #     shear=(-4, 4)
            # ),
            iaa.Fliplr(1.0),  # horizontal flips
            # color jitter, only affects the image
            # iaa.AddToHueAndSaturation((-50, 50))
        ])

    def __call__(self, sample):
        if not ('annot' in sample.keys()):
            smpl = {'img': self.seq(image=sample['img']), "name":sample["name"]}
            return smpl
        image, annots = sample['img'], sample['annot']
        new_annots = annots.copy()
        kps = []
        ###################
        # Alphas = list()
        ##################
        for x, y, alpha in annots[:, 1:NUM_VARIABLES+1]:
            x0, y0, x1, y1 = get_dots(x, y, alpha, distance_thresh=60)
            kps.append(Keypoint(x=x0, y=y0))
            kps.append(Keypoint(x=x1, y=y1))

            ##########################################
            # Alphas.append(alpha)
            # Alphas.append(alpha)
            # cv.circle(sample["img"], (x1, y1), 1, (0, 255, 0))
            ##########################################

        kpsoi = KeypointsOnImage(kps, shape=image.shape)

        image_aug, kpsoi_aug = self.seq(image=image, keypoints=kpsoi)
        #######################################
        imgaug_copy = image_aug.copy()
        # assert(len(Alphas) == (len(kpsoi_aug.keypoints)))
        ###################################
        for i, _ in enumerate(kpsoi_aug.keypoints):
            if i % 2 == 1:
                continue
            kp = kpsoi_aug.keypoints[i]
            x0, y0 = kp.x, kp.y
            kp = kpsoi_aug.keypoints[i+1]
            x1, y1 = kp.x, kp.y

            alpha = get_alpha(x0, y0, x1, y1)
            new_annots[i//2,
            1:NUM_VARIABLES+1] = x0, y0, alpha # abs(180 - alpha)

        x_in_bound = np.logical_and(
            new_annots[:, 0] > 0, new_annots[:, 0] < image_aug.shape[1])
        y_in_bound = np.logical_and(
            new_annots[:, 1] > 0, new_annots[:, 1] < image_aug.shape[0])
        in_bound = np.logical_and(x_in_bound, y_in_bound)

        new_annots = new_annots[in_bound, :]
        ##############################################
        save_dir = "../visualization/aug_imgs"
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, "{}.png".format(np.random.randint(0, 1000)))
        for x, y, alpha in new_annots:
            imgaug_copy = std_draw_line(
                imgaug_copy,
                point=(x, y),
                alpha=alpha,
                mode=DrawMode.Accept
            )
        cv.imwrite(save_path, imgaug_copy)
        ###############################################
        smpl = {'img': image_aug, 'annot': new_annots}

        return smpl

    def augment_image(self, img):
        return self.seq(image=img)

