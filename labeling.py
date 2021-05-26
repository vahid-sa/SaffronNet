import numpy as np
import csv
from retinanet import utils
from retinanet import settings
from retinanet.settings import NAME, X, Y, ALPHA, SCORE, LABEL, TRUTH


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
    return selected_gts


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


def write_boxes(boxes: np.array, path: str, mod: str, class_dict: dict):
    if mod == "active":
        write_mod = "w"
    elif mod == "corrected":
        write_mod = "wb"
    else:
        raise AssertionError("mod can be 'active' or 'corrected'.")
    fileIO =  open(path, mode=mod)
    writer = csv.writer(fileIO, delimiter=",")
    for box in boxes:
        name = format(int(box[NAME]), "03d")
        x, y, alpha, truth = box[[X, Y, ALPHA, TRUTH]]
        label = class_dict[str(int(box[LABEL]))]
        writable_box = (name, x, y, alpha, label, truth)
        writer.writerow(writable_box)
