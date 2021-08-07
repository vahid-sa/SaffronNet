from __future__ import print_function
from os import path as osp
from retinanet.dataloader import CSVDataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
from .settings import NUM_VARIABLES
from retinanet.utils import compute_distance
from utils.visutils import draw_line

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def draw_selected_ignored(loader, detections, output_dir, division=1):
    nms_color_plate = {0: (0, 255, 255), 1: (0, 255, 0)}
    print()
    for i in range(len(loader)):
        if i % division != 0:
            continue
        img_path = loader[i]["path"]
        img = cv.imread(img_path)
        main_detections = np.squeeze(detections['main'][i])
        main_detections = np.concatenate([main_detections, np.ones(shape=(main_detections.shape[0], 1))], axis=1)
        co_detections = np.squeeze(detections['co'][i])
        co_detections = np.concatenate([co_detections, np.zeros(shape=(co_detections.shape[0], 1))], axis=1)
        image_detections = np.concatenate([main_detections, co_detections])
        image_detections = image_detections[image_detections[:, -1].argsort()]

        for j in range(len(image_detections)):
            det = image_detections[j]
            x, y, alpha = det[:3]
            status = det[-1]
            line_color = nms_color_plate[status]
            center_color = (0, 0, 0)
            img = draw_line(
                image=img,
                p=(x, y),
                alpha=90.0 - alpha,
                line_color=line_color,
                center_color=center_color,
                half_line=True,
                line_thickness=3)
        filename = osp.basename(img_path)
        save_path = osp.join(output_dir, filename)
        cv.imwrite(save_path, img)
        print("\rsaved {0}/{1}".format(i, len(loader)), end='')
    print()


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


def convert_results_to_detections(
        results,
        scale: float,
        num_classes,
) -> list:
    detections = [None] * num_classes
    scores, labels, boxes = results
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    boxes = boxes.cpu().numpy()
    if boxes.shape[0] == 0:
        # copy detections to all_detections
        for label in range(num_classes):
            detections[label] = np.zeros((0, NUM_VARIABLES + 1))
        return detections
    # correct boxes for image scale
    boxes /= scale

    # select detections
    image_boxes = boxes
    image_scores = scores
    image_labels = labels
    image_detections = np.concatenate([image_boxes, np.expand_dims(
        image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
    # copy detections to all_detections
    for label in range(num_classes):
        detections[label] = image_detections[image_detections[:, -1] == label, :-1]
    return detections


def _get_detections(dataset, retinanet):
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
    main_all_detections = [None] * len(dataset)
    co_all_detections = [None] * len(dataset)

    retinanet.eval()
    print()
    with torch.no_grad():
        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            data_in_pred_format = data['img'].permute(2, 0, 1).to(device).float().unsqueeze(dim=0)
            results = retinanet(data_in_pred_format)
            main_all_detections[index] = convert_results_to_detections(
                results=results['main'],
                scale=scale,
                num_classes=dataset.num_classes(),
            )
            co_all_detections[index] = convert_results_to_detections(
                results=results['co'],
                scale=scale,
                num_classes=dataset.num_classes(),
            )
            print('\r{}/{}'.format(index + 1, len(dataset)), end='')
    print()
    all_detections = {"main": main_all_detections, "co": co_all_detections}

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


def evaluate(
        generator: CSVDataset,
        retinanet,
        XYd_threshold=10,
        Ad_threshold=25,
        save_path=None,
        write_dir=None,
        division=1,
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
        generator, retinanet)
    main_detections = all_detections["main"]
    all_annotations = _get_annotations(generator)
    if write_dir is not None:
        draw_selected_ignored(loader=generator, detections=all_detections, output_dir=write_dir, division=division)

    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections = main_detections[i][label]
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
        if save_path != None:
            plt.plot(recall, precision)
            # naming the x axis
            plt.xlabel('Recall')
            # naming the y axis
            plt.ylabel('Precision')

            # giving a title to my graph
            plt.title('Precision Recall curve')

            # function to show the plot
            plt.savefig(save_path + '/' + label_name + '_precision_recall.jpg')

    return average_precisions
    # return 0.5
