import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import skimage
import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')

    parser.add_argument(
        '--dataset', help='Dataset type, must be one of csv or coco.')

    parser.add_argument(
        '--csv_classes', help='Path to file containing class list (see readme)')

    parser.add_argument(
        '--csv_anots', help='Path to file containing list of anotations'
    )

    parser.add_argument(
        '--images_dir', help='images base folder'
    )

    parser = parser.parse_args(args)

    if parser.dataset == 'csv':
        dataset = CSVDataset(train_file=parser.csv_anots, class_list=parser.csv_classes, images_dir=parser.images_dir,
                             transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError(
            'Dataset type not understood (must be csv or coco), exiting.')

    print(len(dataset))
    print(image)
    image = dataset.load_image(0)
    plt.imshow((image * 255).astype(np.int32))
    plt.show()


if __name__ == '__main__':
    main()
