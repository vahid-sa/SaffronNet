import numpy as np
from torch._C import dtype
from .losses import prepare, distance
import torch as t
import gc
from .settings import MAX_ANOT_ANCHOR_ANGLE_DISTANCE, MAX_ANOT_ANCHOR_POSITION_DISTANCE


def filter(predictions, scores, min_score=0.5):
    thresh = (scores > min_score)
    return predictions[thresh], scores[thresh]


def nms(predictions, scores, min_score=0.5, max_distance=20):
    """ Apply nms over predictions
        inputs:
            predictions: torch.Tensor (num_anchors, 3)
            scores: torch.Tensor (num_anchors)
            min_scores: int
        return:
            anchors_nms_idx: np.ndarray
    """

    scores_over_thresh = (scores > min_score)
    original_indices = scores_over_thresh.nonzero(as_tuple=True)[0]
    scores = scores[scores_over_thresh]
    predictions = predictions[scores_over_thresh]

    x = predictions[:, 0].cpu()
    gc.collect()
    dx = distance(ax=x, bx=x, large_matrix=True)
    del x
    y = predictions[:, 1].cpu()
    gc.collect()
    dy = distance(ax=y, bx=y, large_matrix=True)
    del y
    gc.collect()
    dxy = t.sqrt(dx*dx + dy*dy)
    co_dxy = t.sqrt(dx*dx + dy*dy)
    del dx, dy
    gc.collect()
    for i in range(dxy.shape[0]):
        filter_row = t.logical_and(-0.01 < dxy[i, :], dxy[i, :] < max_distance)
        filter_row = filter_row.nonzero(as_tuple=True)[0]
        candidate_scores = scores[filter_row]
        if candidate_scores.shape[0] == 0:
            continue
        arg_max = t.argmax(candidate_scores)
        filter_row = t.cat([filter_row[:arg_max], filter_row[arg_max+1:]])
        dxy[:, filter_row] = -1
        co_dxy[:, arg_max] = -1
    try:
        valid_indices = (dxy[0, :] > 0).nonzero(as_tuple=True)[0]
        valid_original_indices = original_indices[valid_indices]
    except IndexError:
        valid_original_indices = []

    try:
        valid_indices = (co_dxy[0, :] > 0).nonzero(as_tuple=True)[0]
        co_valid_original_indices = original_indices[valid_indices]
    except IndexError:
        co_valid_original_indices = []

    return valid_original_indices, co_valid_original_indices
