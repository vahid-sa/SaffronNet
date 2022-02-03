from __future__ import print_function, division
import enum
import sys
import os

import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from utils.visutils import get_alpha, get_dots, normalize_alpha
from torch.utils.data import Dataset

from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import random
import skimage
import cv2 as cv

from PIL import Image
from .settings import NUM_VARIABLES
from retinanet import utils



class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations',
                         'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img, img_name = self.load_image(idx)
        annot = self.load_annotations(idx)

        sample = {'img': img, 'annot': annot, 'name': img_name}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        img_name = image_info['file_name']
        path = os.path.join(self.root_dir, 'images',
                            self.set_name, img_name)
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0, img_name

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, images_dir, image_extension=".jpg", transform=None, save_output_img_directory=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
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
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(
                    csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise(ValueError(
                'invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        self.image_names = list(self.image_data.keys())

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
        annot = self.load_annotations(idx)
        name = os.path.splitext(os.path.basename(self.image_names[idx]))[0]
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        sample['name'] = name
        return sample

    def load_image(self, image_index):
        img = cv.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, NUM_VARIABLES+1))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            ctr_x = a['x']
            ctr_y = a['y']
            alpha = normalize_alpha(90 - a['alpha'])
            # if (x2-x1) < 1 or (y2-y1) < 1:
            #     continue

            annotation = np.zeros((1, NUM_VARIABLES+1))

            annotation[0, 0] = ctr_x
            annotation[0, 1] = ctr_y
            annotation[0, 2] = alpha

            annotation[0, 3] = self.name_to_label(a['class'])
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                if len(row) == 5:
                    img_id, ctr_x, ctr_y, alpha, class_name = row[:5]
                else:
                    raise AssertionError("Annotation format is not correct.")
            except ValueError:
                raise ValueError(
                    'line {}: format should be \'img_file,ctr_x,ctr_y,alpha,class_name\' or \'img_file,,,,,\''.format(
                        line)
                )

            img_file = os.path.join(self.img_dir, img_id + self.ext)
            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (ctr_x, ctr_y, alpha, class_name) == ('', '', '', ''):
                continue

            ctr_x = self._parse(
                float(ctr_x), int, 'line {}: malformed ctr_x: {{}}'.format(line))
            ctr_y = self._parse(
                float(ctr_y), int, 'line {}: malformed ctr_y: {{}}'.format(line))
            alpha = self._parse(
                float(alpha), int, 'line {}: malformed alpha: {{}}'.format(line))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(
                    line, class_name, classes))

            result[img_file].append(
                {'x': ctr_x, 'y': ctr_y, 'alpha': alpha, 'class': class_name})

        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)

    @property
    def num_labels(self):
        f = open(self.train_file, "r")
        reader = csv.reader(f)
        count = sum([1 for _ in reader])
        f.close()
        return count


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    names = [s['name'] for s in data]
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

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones(
            (len(annots), max_num_annots, NUM_VARIABLES+1)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, NUM_VARIABLES+2)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, 'name': names}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    """Resizer: checked!
    """

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

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

        annots[:, :NUM_VARIABLES] *= scale

        return {
            'img': torch.from_numpy(new_image),
            'annot': torch.from_numpy(annots),
            'scale': scale,
        }


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self) -> None:
        super().__init__()
        ia.seed(3)
        self.seq = iaa.Sequential([
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
                shear=(-4, 4)
            ),
            iaa.Fliplr(0.5),  # horizontal flips
            # color jitter, only affects the image
            # iaa.AddToHueAndSaturation((-50, 50))
        ])

    def __call__(self, sample, aug=0.7):

        if np.random.rand() < aug:
            image, annots = sample['img'], sample['annot']

            new_annots = annots.copy()
            kps = []
            for x, y, alpha in annots[:, :NUM_VARIABLES]:
                x0, y0, x1, y1 = get_dots(x, y, alpha, distance_thresh=60)
                kps.append(Keypoint(x=x0, y=y0))
                kps.append(Keypoint(x=x1, y=y1))
            kpsoi = KeypointsOnImage(kps, shape=image.shape)
            image_aug, kpsoi_aug = self.seq(image=image, keypoints=kpsoi)
            for i, _ in enumerate(kpsoi_aug.keypoints):
                if i % 2 == 1:
                    continue
                kp = kpsoi_aug.keypoints[i]
                x0, y0 = kp.x, kp.y
                kp = kpsoi_aug.keypoints[i+1]
                x1, y1 = kp.x, kp.y

                alpha = get_alpha(x0, y0, x1, y1)
                new_annots[i//2,
                           :NUM_VARIABLES] = x0, y0, alpha

            x_in_bound = np.logical_and(
                new_annots[:, 0] > 0, new_annots[:, 0] < image_aug.shape[1])
            y_in_bound = np.logical_and(
                new_annots[:, 1] > 0, new_annots[:, 1] < image_aug.shape[0])
            in_bound = np.logical_and(x_in_bound, y_in_bound)
            new_annots = new_annots[in_bound, :]

            sample = {
                'img': image_aug,
                'annot': new_annots,
            }

        return sample


class Normalizer(object):
    """ Normalizer: checked!
    """

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None, standarded=True):
        self.mean = [0.485, 0.456, 0.406] if (mean is None) else mean
        self.std = [0.229, 0.224, 0.225] if (std is None) else std
        self.coefficient = 255.0 if standarded else 1.0

    def __call__(self, image: (torch.Tensor, np.ndarray)):
        tensor = UnNormalizer._get_tensor(image=image)
        dtype, shape = tensor.dtype, tensor.shape
        tensor = tensor.to(dtype=torch.float64)
        tensor = self._reshape_tensor(tensor=tensor)
        tensor = self._normalize(input_tensor=tensor)
        tensor.mul_(self.coefficient)
        tensor = tensor.to(dtype=dtype)
        tensor = tensor.reshape(shape=shape)
        output = tensor.numpy() if isinstance(image, np.ndarray) else tensor
        return output

    def _normalize(self, input_tensor: torch.Tensor):
        tensor = torch.clone(input_tensor)
        for i in range(tensor.shape[0]):
            tensor[i].mul_(self.std[i])
            tensor[i].add_(self.mean[i])
        return tensor

    def _reshape_tensor(self, tensor: torch.Tensor):
        assert len(self.mean) == len(self.std), "mean and std should be equal shapes"
        assert len(self.mean) in tensor.shape, "input tensor and mean size are not compatible"
        reshaped_tensor = tensor.reshape((len(self.mean), -1))
        return reshaped_tensor

    @staticmethod
    def _get_tensor(image):
        if isinstance(image, torch.Tensor):
            tensor = torch.clone(image)
        elif isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image)
        else:
            raise TypeError("Incorrect input type")
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
