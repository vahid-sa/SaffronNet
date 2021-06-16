"""
Cluster flowers based on assign base_anchors according to angle.
"""
import argparse
from utils import visutils
import torch
from retinanet.dataloader import CSVDataset, Augmenter
from torch.utils import data
import matplotlib.pyplot as plt
from retinanet.settings import ANGLE_SPLIT, NUM_VARIABLES
from retinanet.anchors import generate_anchors, Anchors
import cv2 as cv
from os.path import join
import os
import numpy as np
from retinanet.losses import FocalLoss, calc_distance
from utils.visutils import DrawMode, draw_line, std_draw_line


def setup_parser(args):
    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')

    parser.add_argument(
        '--csv_classes', help='Path to file containing class list (see readme)')

    parser.add_argument(
        '--csv_anots', help='Path to file containing list of anotations')

    parser.add_argument(
        '--images_dir', help='images base folder')

    parser.add_argument(
        '--save_dir', help='output directory for generated images')

    parser = parser.parse_args(args)

    return parser


def cluster_flowers(image, image_name, annots, h=140, w=140, base_dir='/media/mj-haghighi/data/Saffron/Cluster'):
    base_anchors = generate_anchors(ANGLE_SPLIT, NUM_VARIABLES)
    loss = FocalLoss()
    _image = image.copy()
    image = torch.tensor(image)
    image = image.permute(2, 0, 1)
    image = torch.unsqueeze(image, 0)
    anchors = Anchors()(image)[0, :, :]
    dxy, dalpha = calc_distance(
        anchors,
        torch.tensor(annots[:, :NUM_VARIABLES]))
    dxy_min, dxy_argmin = torch.min(dxy, dim=1)
    targets, _, _ = loss.calculate_targets(
        target_shape=(anchors.shape[0], 1),
        dxy_min=dxy_min,
        dxy_argmin=dxy_argmin,
        dalpha=dalpha,
        center_alpha_annotation=torch.tensor(annots))
    image = _image
    for anchor in anchors[targets.squeeze() == 1]:
        x, y, alpha = anchor[0], anchor[1], anchor[2]
        image = draw_line(
            image, (x, y), alpha,
            line_color=(0, 255, 0),
            center_color=(0, 0, 255),
            half_line=True,
            distance_thresh=40,
            line_thickness=2)
    for anchor in anchors[targets.squeeze() == -1]:
        x, y, alpha = anchor[0], anchor[1], anchor[2]
        image = draw_line(
            image, (x, y), alpha,
            line_color=(255, 255, 0),
            center_color=(0, 0, 255),
            half_line=True,
            distance_thresh=40,
            line_thickness=2)

    for i, annot in enumerate(annots):
        x, y, alpha, _, _ = annot
        segment_number = alpha // 22.5

        image = std_draw_line(image, point=(
            x, y), alpha=alpha, mode=DrawMode.Accept)
        image = draw_line(image, p=(x, y),
                          alpha=base_anchors[int(segment_number) % len(base_anchors), 2], line_color=(255, 255, 255), center_color=(255, 255, 255), distance_thresh=60, line_thickness=1, half_line=True)
        image = draw_line(image, p=(x, y),
                          alpha=base_anchors[int(segment_number+1) % len(base_anchors), 2], line_color=(0, 255, 255), center_color=(255, 255, 255), distance_thresh=60, line_thickness=1, half_line=True)

        flower = image[int(max(y-h//2, 0)):int(min(y+h//2, image.shape[0])),
                       int(max(x-w//2, 0)):int(min(x+w//2, image.shape[1]))]

        _dir = join(base_dir, "{}".format(segment_number))
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
        if flower.shape[0] * flower.shape[1] == 0:
            continue
        cv.imwrite(join(_dir, "{}-{}.png".format(image_name, i)), flower)


def main(args=None):
    parser = setup_parser(args)
    augmenter = Augmenter()
    dataset = CSVDataset(
        images_dir=parser.images_dir,
        train_file=parser.csv_anots,
        class_list=parser.csv_classes)

    for image_index in range(len(dataset)):
        image = dataset.load_image(image_index)
        annots = dataset.load_annotations(image_index)
        sample = {'img': image, 'annot': annots}
        for i in range(10):
            sample_aug = augmenter(sample=sample, aug=1)
            cluster_flowers(
                image=sample_aug['img']*255,
                image_name="i{}-a{}".format(image_index, i),
                annots=sample_aug['annot'])


if __name__ == "__main__":
    main()
    # print()
