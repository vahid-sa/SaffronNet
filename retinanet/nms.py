import numpy as np
from torch._C import dtype
from .losses import prepare, distance
import torch as t
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
    y = predictions[:, 1].cpu()
    dx = distance(ax=x, bx=x)
    dy = distance(ax=y, bx=y)
    dxy = t.sqrt(dx*dx + dy*dy)
    del x, y, dx, dy
    for i in range(dxy.shape[0]):
        filter_row = t.logical_and(-0.01 < dxy[i, :], dxy[i, :] < max_distance)
        filter_row = filter_row.nonzero(as_tuple=True)[0]
        if t.cuda.is_available():
            filter_row = filter_row.cuda()
        candidate_scores = scores[filter_row]
        if candidate_scores.shape[0] == 0:
            continue
        arg_max = t.argmax(candidate_scores)
        filter_row = t.cat([filter_row[:arg_max], filter_row[arg_max+1:]])
        dxy[:, filter_row] = -1
    try:
        valid_indices = (dxy[0, :] > 0).nonzero(as_tuple=True)[0]
    except IndexError:
        return []

    return original_indices[valid_indices]
