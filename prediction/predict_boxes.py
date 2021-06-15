# x, y, alpha, label, score
# shape: (N, 5)
# dtype: np.float32

import numpy as np
from retinanet.settings import NAME, SCORE


def select_uncertain_indices(boxes, budget, center_score=0.5):
    scores = np.abs(boxes[:, SCORE] - center_score)
    sort_arguments = scores.argsort()
    selected = sort_arguments[:budget]
    return selected


def select_noisy_indices(boxes, uncertain_selected_indices, previous_corrected_boxes_names, noisy_thresh=0.25):
    scores = boxes[:, SCORE]
    lower_bound = scores < noisy_thresh
    upper_bound = scores > 1 - noisy_thresh
    selected_by_score = np.logical_or(lower_bound, upper_bound)
    selected_by_score[uncertain_selected_indices] = False

    candidate_names = np.unique(
        np.concatenate([boxes[uncertain_selected_indices, NAME], previous_corrected_boxes_names], axis=0))
    selected_by_name = np.in1d(boxes[:, NAME], candidate_names)

    selected = np.logical_and(selected_by_name, selected_by_score)
    selected = np.squeeze(np.argwhere(selected))
    return selected


def split_uncertain_and_noisy(boxes, previous_corrected_boxes_names, budget=100):
    uncertain_indices = select_uncertain_indices(boxes, budget=budget)
    noisy_indices = select_noisy_indices(
        boxes=boxes,
        uncertain_selected_indices=uncertain_indices,
        previous_corrected_boxes_names=previous_corrected_boxes_names,
    )
    uncertain_boxes = boxes[uncertain_indices]
    noisy_boxes = boxes[noisy_indices]
    # status = np.full(shape=(len(boxes), 1), dtype=np.float64, fill_value=-1)
    # status[uncertain_indices] = 0
    # status[noisy_indices] = 1
    return uncertain_boxes, noisy_boxes
    # return np.concatenate((boxes, status), axis=1)
