import torch
import numpy as np
import torch.nn as nn
from .settings import ANGLE_SPLIT, NUM_VARIABLES, STRIDE
from .anchor_utils import *


class Anchors(nn.Module):
    def __init__(self):
        super(Anchors, self).__init__()

    def forward(self, image):

        image_shape = image.shape[2:]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, NUM_VARIABLES)).astype(np.float32)

        anchors = generate_anchors(
            angle_split=ANGLE_SPLIT, num_variables=NUM_VARIABLES)
        shifted_anchors = shift(image_shape, STRIDE, anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))
