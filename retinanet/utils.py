import csv
from enum import Enum

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(
                    np.array([0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(
                    np.array([0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(
                    np.array([1, 1, 1]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(
                    np.array([1, 1, 1]).astype(np.float32))
        else:
            self.std = std

    def forward(self, center_alphas, deltas):

        ctr_x = center_alphas[:, :, 0]
        ctr_y = center_alphas[:, :, 1]
        alpha = center_alphas[:, :, 2]

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dalpha = deltas[:, :, 2] * self.std[2] + self.mean[2]

        pred_ctr_x = ctr_x + dx
        pred_ctr_y = ctr_y + dy
        pred_alpha = alpha + dalpha

        pred_boxes = torch.stack([pred_ctr_x, pred_ctr_y, pred_alpha], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, center_alphas, img):
        batch_size, num_channels, height, width = img.shape

        center_alphas[:, :, 0] = torch.clamp(center_alphas[:, :, 0], min=0)
        center_alphas[:, :, 1] = torch.clamp(center_alphas[:, :, 1], min=0)

        center_alphas[:, :, 0] = torch.clamp(center_alphas[:, :, 0], max=width)
        center_alphas[:, :, 1] = torch.clamp(
            center_alphas[:, :, 1], max=height)

        return center_alphas


def prepare(a, b):
    # extend as cols
    repetitions = b.shape[0]
    at = np.transpose([a] * repetitions)
    # extend as rows
    repetitions = a.shape[0]
    bt = np.tile(b, (repetitions, 1))
    return at, bt


def distance(ax, bx):
    """
    ax: (N) ndarray of float
    bx: (K) ndarray of float
    Returns
    -------
    (N, K) ndarray of distance between all x in ax, bx
    """
    ax, bx = prepare(ax, bx)
    return np.abs(ax - bx)


def compute_distance(a, b) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    a: (N, 3) ndarray of float
    b: (K, 3) ndarray of float
    Returns
    -------
    distances: (N, K) ndarray of distance between center_alpha and query_center_alpha
    """
    ax = a[:, 0]
    bx = b[:, 0]

    ay = a[:, 1]
    by = b[:, 1]

    aa = a[:, 2]
    ba = b[:, 2]

    dalpha = distance(ax=aa, bx=ba)
    dx = distance(ax=ax, bx=bx)
    dy = distance(ax=ay, bx=by)
    dxy = np.sqrt(dx * dx + dy * dy)

    return dxy, dalpha


def load_classes(csv_class_list_path: str) -> Tuple[dict, dict]:
    """
    loads class list defined in dataset
    :param csv_class_list_path: path to csv class list
    :return: a dict that converts class to index and a dict that convert index to class
    """
    index_to_class = dict()
    class_to_index = dict()
    fileIO = open(csv_class_list_path, "r")
    reader = csv.reader(fileIO, delimiter=",")
    for row in reader:
        class_name, str_class_index = row
        index_to_class[str_class_index] = class_name
        class_to_index[class_name] = str_class_index
    fileIO.close()
    return class_to_index, index_to_class


class ActiveLabelMode(Enum):
    noisy = 0
    corrected = 1
    gt = 3
    uncertain = 2
    ignored = 4


class ActiveLabelModeSTR(Enum):
    noisy = "noisy"
    gt = "ground_truth"
    uncertain = "uncertain"
    corrected = "corrected"
    ignored = "ignored"
