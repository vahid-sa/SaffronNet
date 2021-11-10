# x, y, alpha, label, score
# shape: (N, 5)
# dtype: np.float32

import numpy as np
from retinanet.settings import NAME, SCORE
import retinanet


def select_uncertain_indices(boxes, center_score=0.5):
    budget = retinanet.settings.NUM_QUERIES
    print(f"budget: {budget}")
    scores = np.abs(boxes[:, SCORE] - center_score)
    sort_arguments = scores.argsort()
    selected = sort_arguments[:budget]
    return selected


def select_noisy_indices(boxes, uncertain_selected_indices, previous_corrected_boxes_names):
    noisy_thresh = retinanet.settings.NOISY_THRESH
    print(f"noisy_thresh: {noisy_thresh}")
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


def split_uncertain_and_noisy(boxes, previous_corrected_boxes_names):
    uncertain_indices = select_uncertain_indices(boxes)
    noisy_indices = select_noisy_indices(
        boxes=boxes,
        uncertain_selected_indices=uncertain_indices,
        previous_corrected_boxes_names=previous_corrected_boxes_names,
    )
    uncertain_boxes = boxes[uncertain_indices]
    noisy_boxes = boxes[noisy_indices]
    return uncertain_boxes, noisy_boxes
