# x, y, alpha, label, score
# shape: (N, 5)
# dtype: np.float32

import torch
import torchvision
import cv2
import numpy as np
import csv
import time
from os import path as osp
import argparse
import json
from retinanet.settings import X, Y, ALPHA, LABEL, SCORE, TRUTH
from retinanet.utils import load_classes
from retinanet import imageloader
from detector import get_detections
from visualize import draw


def select_uncertain_indices(boxes, budget, center_score=0.5):
    scores = np.abs(boxes[:, SCORE] - center_score)
    sort_arguments = scores.argsort()
    selected = sort_arguments[:budget]
    return selected


def select_noisy_indices(boxes, uncertain_selected_indices, noisy_thresh=0.25):
    scores = boxes[:, SCORE]
    lower_bound = scores < noisy_thresh
    upper_bound = scores > 1 - noisy_thresh
    selected = np.logical_or(lower_bound, upper_bound)
    selected[uncertain_selected_indices] = False
    selected = np.squeeze(np.argwhere(selected))
    return selected


def split(boxes, budget=100):
    uncertain_indices = select_uncertain_indices(boxes, budget=budget)
    noisy_indices = select_noisy_indices(boxes, uncertain_indices)
    uncertain_boxes = boxes[uncertain_indices]
    noisy_boxes = boxes[noisy_indices]
    status = np.full(shape=(len(boxes), 1), dtype=np.float64, fill_value=-1)
    status[uncertain_indices] = 0
    status[noisy_indices] = 1
    # return uncertain_boxes, noisy_boxes
    return np.concatenate((boxes, status), axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predicting boxes and split uncertain and noisy')
    parser.add_argument(
        "--filenames_path", help="filenames")
    parser.add_argument(
        "--partition", help="supervised | unsupervised | validation | test")
    parser.add_argument(
        '--image_dir', help='Path to directory containing images')
    parser.add_argument(
        '--model_path', help='Path to model')
    parser.add_argument(
        '--class_list', help='Path to CSV file listing class names (see README)')
    parser.add_argument(
        "--ext", help="image extrension", default=".jpg")
    parser.add_argument("--output_dir", help="where to save")
    parser = parser.parse_args()

    assert parser.partition in "supervised | unsupervised | validation | test"
    loader = imageloader.CSVDataset(
        filenames_path=parser.filenames_path,
        partition=parser.partition,
        class_list=parser.class_list,
        images_dir=parser.image_dir,
        image_extension=parser.ext,
        transform=torchvision.transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
    )
    retinanet = torch.load(parser.model_path)
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    bboxes = get_detections(loader, retinanet)
    # uncertain_boxes, noisy_boxes = split(bboxes)
    bboxes = split(bboxes)
    # print("uncertain: shape: {0}, dtype: {1}, min_score: {2}, max_score: {3}".format(
    #     uncertain_boxes.shape, uncertain_boxes.dtype, round(uncertain_boxes[:, SCORE].min(), 2), round(uncertain_boxes[:, SCORE].max(), 2)))
    # print("noisy: shape: {0}, dtype: {1}, min_score: {2}, max_score: {3}".format(
    #     noisy_boxes.shape, noisy_boxes.dtype, round(noisy_boxes[:, SCORE].min(), 2), round(noisy_boxes[:, SCORE].max(), 2)))
    draw(loader=loader, detections=bboxes, images_dir=parser.image_dir, output_dir=parser.output_dir)
