import numpy as np
from .losses import __prepare


def nms(predictions, scores, min_score):
    """ Apply nms over predictions
        inputs: 
            predictions: torch.Tensor
            scores: torch.Tensor
            min_scores: int
        return:
            anchors_nms_idx: np.ndarray
    """
    print(predictions.type)
    print(scores.type)
    # r_score, c_score1 = __prepare(scores)
    # scores = r_score - c_score1

    # best = []

    # valid_scores_indices = scores > min_score
    # for i in range(predictions.shape[0]):
