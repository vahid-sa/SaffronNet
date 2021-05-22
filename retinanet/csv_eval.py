from __future__ import print_function
from retinanet.dataloader import CSVDataset
import numpy as np
import cv2 as cv
from utils.visutils import draw_line
import matplotlib.pyplot as plt
import json
import os
import matplotlib.pyplot as plt
import torch
from .settings import MAX_ANOT_ANCHOR_ANGLE_DISTANCE, MAX_ANOT_ANCHOR_POSITION_DISTANCE, NUM_VARIABLES


def prepare(a, b):
    # extend as cols
    repetitions = b.shape[0]
    at = np.transpose([a] * repetitions)
    # extend as rows
    repetitions = a.shape[0]
    bt = np.tile(b, (repetitions, 1))
    return at, bt


def distance(ax, bx):
    """ 
    ax: (N) ndarray of float
    bx: (K) ndarray of float
    Returns
    -------
    (N, K) ndarray of distance between all x in ax, bx
    """
    ax, bx = prepare(ax, bx)
    return np.abs(ax - bx)


def compute_distance(a, b):
    """
    Parameters
    ----------
    a: (N, 3) ndarray of float
    b: (K, 3) ndarray of float
    Returns
    -------
    distances: (N, K) ndarray of distance between center_alpha and query_center_alpha
    """
    ax = a[:, 0]
    bx = b[:, 0]

    ay = a[:, 1]
    by = b[:, 1]

    aa = a[:, 2]
    ba = b[:, 2]

    dalpha = distance(ax=aa, bx=ba)
    dx = distance(ax=ax, bx=bx)
    dy = distance(ax=ay, bx=by)
    dxy = np.sqrt(dx*dx + dy*dy)

    return dxy, dalpha


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


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


def visualize_predictions(img, acc_pred, dec_pred, anots):
    ACC = {
        'LINE': (0, 255, 0),
        'CENTER': (0, 0, 255)
    }
    DEC = {
        'LINE': (255, 0, 0),
        'CENTER': (0, 0, 255)
    }
    ANOT = {
        'LINE': (0, 0, 0),
        'CENTER': (255, 255, 555)
    }

    for a in anots:
        a = a[:NUM_VARIABLES]
        x, y, alpha = a
        img = draw_line(
            image=img,
            p=(x, y),
            alpha=alpha,
            line_color=ANOT['LINE'],
            center_color=ANOT['CENTER'],
            line_thickness=3
        )

    for p in acc_pred:
        p = p[:NUM_VARIABLES]
        x, y, alpha = p
        img = draw_line(
            image=img,
            p=(x, y),
            alpha=alpha,
            line_color=ACC['LINE'],
            center_color=ACC['CENTER'],
            line_thickness=3
        )

    for p in dec_pred:
        p = p[:NUM_VARIABLES]
        x, y, alpha = p
        img = draw_line(
            image=img,
            p=(x, y),
            alpha=alpha,
            line_color=DEC['LINE'],
            center_color=DEC['CENTER'],
            line_thickness=3
        )
    return img


def evaluate(
    generator: CSVDataset,
    retinanet,
    XYd_threshold=10,
    Ad_threshold=25,
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
            num_annotations += annotations.shape[0]
            detected_annotations = []
            acc_pred = []
            dec_pred = []
            for d in detections:
                scores = np.append(scores, d[NUM_VARIABLES])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                dxys, dangels = compute_distance(
                    np.expand_dims(d, axis=0), annotations)

                assigned_annotation = np.argmin(dxys, axis=1)
                min_dxy = dxys[0, assigned_annotation]
                min_dangel = dangels[0, assigned_annotation]

                if min_dxy <= XYd_threshold and min_dangel <= Ad_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                    acc_pred.append(d)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    dec_pred.append(d)
            img = generator.load_image(i)
            img = (img * 255).astype(np.int32)
            img = visualize_predictions(img, acc_pred, dec_pred, annotations)
            # cv.imshow('validation', img)
            plt.imshow(img)
            plt.show()
        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / \
            np.maximum(true_positives + false_positives,
                       np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    print('\nmAP:')
    for label in range(generator.num_classes()):
        label_name = generator.label_to_name(label)
        print('{}: {}'.format(label_name, average_precisions[label][0]))
        if average_precisions[label][0] < 0.01:
            continue
        print("Precision: ", precision[-1])
        print("Recall: ", recall[-1])

        if save_path != None:
            plt.plot(recall, precision)
            # naming the x axis
            plt.xlabel('Recall')
            # naming the y axis
            plt.ylabel('Precision')

            # giving a title to my graph
            plt.title('Precision Recall curve')

            # function to show the plot
            plt.savefig(save_path+'/'+label_name+'_precision_recall.jpg')

    return average_precisions
    # return 0.5
