from __future__ import print_function

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
from torch.cuda.random import get_rng_state
from .settings import MAX_ANOT_ANCHOR_ANGLE_DISTANCE, MAX_ANOT_ANCHOR_POSITION_DISTANCE, NUM_VARIABLES
import math

# def prepare(a, b):
#     # extend as cols
#     repetitions = b.shape[0]
#     at = np.transpose([a] * repetitions)
#     # extend as rows
#     repetitions = a.shape[0]
#     bt = np.tile(b, (repetitions, 1))
#     return at, bt


# def distance(ax, bx):
#     """
#     ax: (N) ndarray of float
#     bx: (K) ndarray of float
#     Returns
#     -------
#     (N, K) ndarray of distance between all x in ax, bx
#     """
#     ax, bx = prepare(ax, bx)
#     return np.abs(ax - bx)


# def compute_distance(a, b):
#     """
#     Parameters
#     ----------
#     a: (N, 3) ndarray of float
#     b: (K, 3) ndarray of float
#     Returns
#     -------
#     distances: (N, K) ndarray of distance between center_alpha and query_center_alpha
#     """
#     ax = a[:, 0]
#     bx = b[:, 0]

#     ay = a[:, 1]
#     by = b[:, 1]

#     aa = a[:, 2]
#     ba = b[:, 2]

#     dalpha = distance(ax=aa, bx=ba)
#     dx = distance(ax=ax, bx=bx)
#     dy = distance(ax=ay, bx=by)
#     dxy = np.sqrt(dx*dx + dy*dy)

#     return dxy, dalpha


# def _compute_ap(recall, precision):
#     """ Compute the average precision, given the recall and precision curves.
#     Code originally from https://github.com/rbgirshick/py-faster-rcnn.
#     # Arguments
#         recall:    The recall curve (list).
#         precision: The precision curve (list).
#     # Returns
#         The average precision as computed in py-faster-rcnn.
#     """
#     # correct AP calculation
#     # first append sentinel values at the end
#     mrec = np.concatenate(([0.], recall, [1.]))
#     mpre = np.concatenate(([0.], precision, [0.]))

#     # compute the precision envelope
#     for i in range(mpre.size - 1, 0, -1):
#         mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

#     # to calculate area under PR curve, look for points
#     # where X axis (recall) changes value
#     i = np.where(mrec[1:] != mrec[:-1])[0]

#     # and sum (\Delta recall) * prec
#     ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#     return ap


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(
        dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()

    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = retinanet(data['img'].permute(
                    2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = retinanet(
                    data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(
                    image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros(
                        (0, NUM_VARIABLES+1))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(
        generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, NUM_VARIABLES]
                                                    == label, :NUM_VARIABLES].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def distance(x0, y0, x1, y1):
    return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)


def _calculate_accuracy_for_each_prediction(anot, prediction):
    x, y, alpha = anot
    px, py, palpha = prediction
    d = distance(x, y, px, py)
    dalpha = abs(alpha - palpha)
    if d < MAX_ANOT_ANCHOR_POSITION_DISTANCE and dalpha < MAX_ANOT_ANCHOR_ANGLE_DISTANCE:
        return (1 - float(d)/MAX_ANOT_ANCHOR_POSITION_DISTANCE) * math.cos(dalpha * (math.pi / 180.0))
        # return 1
    return 0


def calculate_metrics(anots, predictions):
    """ calculate accuracy for each image
      inputs: 
        anots: (num_anchors, 3)
        predictions: (num_anchors, 4)
        anchors: (num_anchors, 3)
    """
    n, k = len(predictions), len(anots)
    map = np.zeros((n, k))

    for _n, prediction in enumerate(predictions):
        for _k, anot in enumerate(anots):
            map[_n, _k] = _calculate_accuracy_for_each_prediction(
                anot, prediction)

    predictions_max_score = np.amax(map, axis=1)
    anots_max_score = np.amax(map, axis=0)

    TP = np.sum(predictions_max_score > 0.001)
    FP = np.sum(predictions_max_score < 0.001)
    FN = np.sum(anots_max_score < 0.001)

    return TP, FP, FN


def evaluate(
    generator,
    retinanet,
    XYd_threshold=10 * MAX_ANOT_ANCHOR_POSITION_DISTANCE,
    Ad_threshold=MAX_ANOT_ANCHOR_ANGLE_DISTANCE,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save precision recall curve of each label.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations

    all_detections = _get_detections(
        generator, retinanet, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations = _get_annotations(generator)

    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            print('detections.shape: ', detections.shape)
            print('annotations.shape: ', annotations.shape)

    # return average_precisions
    return 0.5
