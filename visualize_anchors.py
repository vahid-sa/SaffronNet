from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer

from retinanet.anchors import Anchors
from utils.visutils import write_angle, draw_line
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch
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
import cv2 as cv


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

    parser.add_argument(
        '--save_dir', help='output directory for generated images'
    )

    parser = parser.parse_args(args)

    if parser.dataset == 'csv':
        dataset = CSVDataset(train_file=parser.csv_anots, class_list=parser.csv_classes, images_dir=parser.images_dir,
                             transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError(
            'Dataset type not understood (must be csv or coco), exiting.')

    # for i in range(len(dataset)):
    #     image = (dataset.load_image(i) * 255).astype(np.int32)
    #     anots = dataset.load_annotations(i)
    #     for anot in anots:
    #         x, y, alpha = anot[0], anot[1], 90 - anot[2]
    #         image = draw_line(
    #             image, (x, y), alpha,
    #             line_color=(0, 0, 0),
    #             center_color=(255, 0, 0),
    #             half_line=True
    #         )

    #     image_name = os.path.basename(dataset.image_names[i])
    #     cv.imwrite(os.path.join(parser.save_dir, image_name),
    #                cv.cvtColor(image, cv.COLOR_RGB2BGR))

    sample_image = (dataset.load_image(0) * 255).astype(np.int32)
    sample_batch = np.expand_dims(sample_image, axis=0)
    sample_batch = sample_batch.transpose(0, 3, 1, 2)
    print(sample_batch.shape)
    anchros_mudole = Anchors()
    anchors = anchros_mudole(sample_batch)
    print(anchors)


if __name__ == '__main__':
    main()
