import numpy as np
from torch._C import dtype
from .losses import prepare, distance
import torch as t
from .settings import MAX_ANOT_ANCHOR_ANGLE_DISTANCE, MAX_ANOT_ANCHOR_POSITION_DISTANCE


def nms(predictions, scores, min_score, max_distance=20):
    """ Apply nms over predictions
        inputs: 
            predictions: torch.Tensor (num_anchors, 3)
            scores: torch.Tensor (num_anchors)
            min_scores: int
        return:
            anchors_nms_idx: np.ndarray
    """
    arg_bests = set()
    x = predictions[:, 0]
    y = predictions[:, 1]
    dx = distance(ax=x, bx=x)
    dy = distance(ax=y, bx=y)
    dxy = t.sqrt(dx*dx + dy*dy)
    all_adj_indices = dxy < max_distance

    I = t.diag(t.ones(predictions.shape[0]) * -1) + 1
    I = I.bool()
    if t.cuda.is_available():
        I = I.cuda()
    all_adj_indices = all_adj_indices * I  # to filter diag
    for i in range(all_adj_indices.shape[0]):
        adj_indices = all_adj_indices[i, :]
        adj_args = adj_indices.nonzero(as_tuple=True)[0]

        candidate_scores = scores[adj_args]
        if candidate_scores.nelement() == 0:
            continue
        max_score_arg = adj_args[t.argmax(candidate_scores)]

        best_prediction_arg = max_score_arg
        arg_bests.add(best_prediction_arg.tolist())  # from tensor to int
    return t.Tensor(list(arg_bests)).long()
