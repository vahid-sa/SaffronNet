import numpy as np
import csv
from retinanet import utils
from retinanet import settings
from retinanet.settings import NAME, X, Y, ALPHA, SCORE, LABEL, TRUTH


def active_status_int_to_string(int_status: float) -> utils.ActiveLabelModeSTR:
    if int_status == utils.ActiveLabelMode.corrected.value:
        str_status = utils.ActiveLabelModeSTR.corrected.value
    elif int_status == utils.ActiveLabelMode.noisy.value:
        str_status = utils.ActiveLabelModeSTR.noisy.value
    else:
        raise AssertionError("truth_status field can be '0' or '1' ")
    return str_status


def label(all_gts, all_uncertain_preds):
    """
    :param gt: [[filename, x, y, alpha, label]]
    :param all_uncertain_preds: [[filename, x, y, alpha, score, label]]
    :return: [[filename, x, y, alpha, label]]
    """
    names = np.unique(all_uncertain_preds[:, NAME])
    selected_gts = list()
    for name in names:
        preds = all_uncertain_preds[all_uncertain_preds[:, NAME] == name]
        gts = all_gts[all_gts[:, NAME] == name]
        dxy, dalpha = utils.compute_distance(preds, gts)

        gts_to_preds_index_candidates, assined_gts_to_preds_candidates = dxy.argmin(axis=1), dxy.min(axis=1)
        assigned_gts_to_preds_indices = np.unique(
            gts_to_preds_index_candidates[assined_gts_to_preds_candidates < settings.MAX_CORRECTABLE_DISTANCE])
        assined_annotations = gts[assigned_gts_to_preds_indices].tolist()
        selected_gts.extend(assined_annotations)
    return np.asarray(selected_gts, dtype=np.float32)


def filter_noisy_by_asked_box_images(noisy_boxes, valid_imagenames):
    selected_noisy_boxes = noisy_boxes[np.in1d(noisy_boxes[:, NAME], valid_imagenames)]
    return selected_noisy_boxes


def insert_status(boxes, truth):
    status = np.full(shape=(len(boxes)), dtype=boxes.dtype, fill_value=truth)
    out_boxes = np.zeros(shape=(boxes.shape[0], boxes.shape[1] + 1), dtype=boxes.dtype)
    out_boxes[:, [NAME, X, Y, ALPHA, SCORE, LABEL]] = boxes[:, [NAME, X, Y, ALPHA, SCORE, LABEL, TRUTH]]
    out_boxes[:, TRUTH] = status
    return out_boxes


def merge_noisy_and_asked(corrected_boxes, noisy_boxes):
    filtered_noisy_boxes = filter_noisy_by_asked_box_images(noisy_boxes, corrected_boxes[:, NAME])
    corrected_boxes_with_status = insert_status(corrected_boxes, truth=1)
    noisy_boxes_with_status = insert_status(filtered_noisy_boxes, truth=0)
    all_boxes = np.concatenate((corrected_boxes_with_status, noisy_boxes_with_status), axis=0)
    sorted_all_boxes = all_boxes[all_boxes[:, NAME].argsort()]
    return sorted_all_boxes


def write_active_boxes(boxes: np.array, path: str, class_dict: dict):
    assert (len(boxes.shape) == 2) and (
                boxes.shape[1] == len([NAME, X, Y, ALPHA, LABEL, TRUTH])), "Incorrect boxes format."
    fileIO = open(path, mode="w")
    writer = csv.writer(fileIO, delimiter=",")
    for box in boxes:
        name = format(int(box[NAME]), "03d")
        x, y, alpha, ground_truth_status = box[[X, Y, ALPHA, TRUTH]]
        label = class_dict[str(int(box[LABEL]))]
        status = active_status_int_to_string(int_status=ground_truth_status)
        writable_box = (name, x, y, alpha, label, status)
        writer.writerow(writable_box)
    fileIO.close()


def write_corrected_boxes(boxes: np.array, path: str, class_dict: dict):
    assert (len(boxes.shape) == 2) and (boxes.shape[1] == len([NAME, X, Y, ALPHA, LABEL])), "Incorrect boxes format."
    fileIO = open(path, mode="a")
    writer = csv.writer(fileIO, delimiter=",")
    for box in boxes:
        name = format(int(box[NAME]), "03d")
        x, y, alpha = box[[X, Y, ALPHA]]
        label = class_dict[str(int(box[LABEL]))]
        writable_box = (name, x, y, alpha, label)
        writer.writerow(writable_box)
    fileIO.close()
