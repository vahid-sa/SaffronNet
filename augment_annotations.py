import os
import numpy as np
import torch
import torchvision
import cv2 as cv
import shutil
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import csv
import random
import argparse
from os import path as osp
import copy
from utils.visutils import DrawMode, get_alpha, get_dots, std_draw_line, draw_line
from retinanet.dataloader import CSVDataset

NUM_VARIABLES = 3
aug_detection_number = 0


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
            iaa.Fliplr(1),  # horizontal flips
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
            imgaug_copy = image_aug.copy()
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
            for x, y, alpha, _, _ in new_annots:
                imgaug_copy = std_draw_line(
                    imgaug_copy,
                    point=(x, y),
                    alpha=alpha,
                    mode=DrawMode.Accept
                )
            sample = {'img': image_aug, 'annot': new_annots}

        return sample


def draw(img, boxes, augmented=False):
    im = img.copy()
    for box in boxes:
        x, y, alpha = box[0:NUM_VARIABLES]
        p = (x, y)
        center_color = (0, 0, 0)
        if augmented:
            line_clor = (0, 0, 255)
        else:
            line_clor = (255, 0, 0)
        im = draw_line(image=im, p=p, alpha=alpha, line_color=line_clor, center_color=center_color, half_line=True)
    return im


aug = Augmenter()
train_file = "annotations/supervised.csv"
class_list = "annotations/labels.csv"
images_dir = "dataset/Train"
dataloader = CSVDataset(train_file=train_file, class_list=class_list, images_dir=images_dir)
img_path = dataloader.image_names[3]
img = cv.imread(img_path)
annotations = dataloader[3]['annot']
sample = aug(sample={'img': img, 'annot': annotations})
sample = aug(sample=sample)
aug_img, aug_annots = sample["img"], sample["annot"]

aug_img = draw(img=aug_img, boxes=aug_annots)
cv.imshow("img", aug_img)
cv.waitKey(0)
cv.destroyAllWindows()
