from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer
from retinanet.losses import calc_distance
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
from retinanet.settings import *
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

    sample_image = (dataset.load_image(0) * 255).astype(np.int32)
    sample_batch = np.expand_dims(sample_image, axis=0)
    sample_batch = sample_batch.transpose(0, 3, 1, 2)
    anchros_mudole = Anchors()
    anchors = anchros_mudole(sample_batch)

    for i in range(len(dataset)):
        image = (dataset.load_image(i) * 255).astype(np.int32)
        anots = dataset.load_annotations(i)

        targets = torch.ones((anchors.shape[1], 1)) * -1

        distance, deltaphi = calc_distance(torch.tensor(anchors[0, :, :]),
                                           torch.tensor(anots[:, :NUM_VARIABLES]))
        distance_min, distance_argmin = torch.min(
            distance, dim=1)  # num_anchors x 1
        deltaphi_min, deltaphi_argmin = torch.min(
            deltaphi, dim=1)  # num_anchors x 1

        targets[torch.ge(
            distance_min, 1.5 * MAX_ANOT_ANCHOR_POSITION_DISTANCE), :] = 0
        targets[torch.ge(
            deltaphi_min, 2 * MAX_ANOT_ANCHOR_ANGLE_DISTANCE), :] = 0

        positive_indices = torch.logical_and(
            torch.le(distance_min, MAX_ANOT_ANCHOR_POSITION_DISTANCE),
            torch.le(deltaphi_min, MAX_ANOT_ANCHOR_ANGLE_DISTANCE))

        num_positive_anchors = positive_indices.sum()

        # assigned_annotations = center_alpha_annotation[deltaphi_argmin, :] # no different in result
        assigned_annotations = anots[distance_argmin, :]

        targets[positive_indices, :] = 0
        targets[positive_indices,
                assigned_annotations[positive_indices, 3]] = 1

        anchors = anchors[0, :, :]
        for anchor in anchors[targets.squeeze() == 1]:
            x, y, alpha = anchor[0], anchor[1], anchor[2]
            image = draw_line(
                image, (x, y), alpha,
                line_color=(0, 255, 0),
                center_color=(0, 0, 255),
                half_line=True,
                distance_thresh=60
            )
        for anot in anots:
            x, y, alpha = anot[0], anot[1], 90 - anot[2]
            image = draw_line(
                image, (x, y), alpha,
                line_color=(0, 0, 0),
                center_color=(255, 0, 0),
                half_line=True
            )
        image_name = os.path.basename(dataset.image_names[i])
        print(image.shape)
        print(image.dtype)
        cv.imwrite(os.path.join(parser.save_dir, image_name),
                   cv.cvtColor(image, cv.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
