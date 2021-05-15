# x, y, alpha, label, score
# shape: (N, 5)
# dtype: np.float32

import torch
import cv2
import numpy as np
import csv
import time
from os import path as osp
import argparse
import json
from retinanet.settings import X, Y, ALPHA, LABEL, SCORE, TRUTH


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def detect_image(image_dir, filenames, model_path, class_list, ext=".jpg"):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))
    labels = {}
    for key, value in classes.items():
        labels[value] = key
    print("labels", labels)
    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()
    bboxes = list()
    for img_name in filenames:

        image = cv2.imread(osp.join(image_dir, img_name+ext))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)

        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))
        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()

            scores, classification, transformed_anchors = model(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
        scores = transformed_anchors.cpu().detach().numpy()
        classification = classification.cpu().detach().numpy()
        transformed_anchors = transformed_anchors.cpu().detach().numpy()
        # print("scores: {0}   class: {1}   anchors: {2}".format(scores.shape, classification.shape, transformed_anchors.shape))
        # print("scores: {0}   class: {1}   anchors: {2}".format(scores.dtype, classification.dtype, transformed_anchors.dtype))
        print(scores[0])
        print(np.unique(classification))
        # bboxes.extend(np.concatenate((transformed_anchors, classification, scores), axis=1))
    return bboxes


# def sort_by_scores(bboxes):
#     sorted_indices = bboxes[:, SCORE].argsort()
#     sorted_bboxes = bboxes[sorted_indices]
#     return sorted_bboxes

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
    return uncertain_boxes, noisy_boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predicting boxes and split uncertain and noisy')
    parser.add_argument("--image_dir", help="where images are saved")
    parser.add_argument("--model_path", help="path to weights")
    parser = parser.parse_args()
    image_dir = parser.image_dir
    model_path = parser.model_path
    with open("annotations/filenames.json", "r") as fileIO:
        str_names = fileIO.read()
    filenames = json.loads(str_names)
    bboxes = detect_image(image_dir=image_dir, filenames=filenames["unsupervised"], model_path=model_path, class_list="annotations/labels.csv")
    print(bboxes)
    print(bboxes.shape)
    print(bboxes.dtype)
