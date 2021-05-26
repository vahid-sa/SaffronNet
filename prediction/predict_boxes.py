# x, y, alpha, label, score
# shape: (N, 5)
# dtype: np.float32

import numpy as np
from retinanet.settings import SCORE


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
