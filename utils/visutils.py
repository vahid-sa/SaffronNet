from argparse import RawDescriptionHelpFormatter
import math
from enum import Enum
import numpy as np
import logging
from typing import Tuple
from cv2 import cv2 as cv
from os import path as osp
from retinanet.settings import ACC, DEC, RAW
POINT = Tuple[float, float]
COLOR = Tuple[int, int, int]


class DrawMode(Enum):
    Accept = 1
    Decline = 2
    Raw = 3


def _y(m, x, b):
    return b - (m * x)


def _b(x, y, m):
    return y + (m * x)


def distance(x0, y0, x1, y1):
    return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)


def write_angle(
    image: np.ndarray,
    p: POINT,
    alpha: float,
    color: COLOR = (0, 255, 0),
    y_bias: int = 10,
    x_bias: int = 10
) -> np.ndarray:
    """ Write angle beside saffron
        inputs: 
            image: input image 
            p: center of saffron 
            alpha: angle of saffron
            color: color of text
            y_bias: distance of text from saffron -y
            x_bias: distance of text from saffron -x
        return:
            image with text: (np.ndarray)
    """
    x = p[0]
    y = p[1]
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(image, str(alpha), (int(x)+x_bias, int(y)+y_bias),
               font, 0.8, color, 2, cv.LINE_AA)
    return image


def get_dots(x, y, alpha, distance_thresh=20, m_thresh=100, ignore_center=False):
    """ Calculte (x1, y1, x2 , y2) from (x, y, alpha)
        inputs: 
            x: x pos
            y: y pos
            alpha: saffron angle

        return: 

    """
    if alpha == 0:
        alpha += 0.001

    m = math.tan(alpha * (math.pi / 180))
    m = max(min(m, m_thresh), -m_thresh)

    b = _b(x, y, m)

    sign = 1 if math.cos(alpha * (math.pi / 180)) > 0 else -1
    new_x = x + sign*20
    new_y = _y(m, new_x, b)
    while distance(x, y, new_x, new_y) > distance_thresh:
        new_x = x + sign*(abs(x - new_x) / 2)
        new_y = _y(m, new_x, b)

    if ignore_center:
        _new_x = x - sign*abs(x - new_x)
        _new_y = _y(m, _new_x, b)
    else:
        _new_x = x
        _new_y = y

    return _new_x, _new_y, new_x, new_y


def draw_line(
    image: np.ndarray,
    p: POINT,
    alpha: float,
    line_color: COLOR = None,
    center_color: COLOR = None,
    distance_thresh: int = 30,
    line_thickness: int = 5,
    center_thickness: int = 2,
    half_line: bool = False
):
    """ Draw line based on annotation
        input: 
            image: input image 
            p: center of saffron 
            alpha: angle of saffron
            line_color: colr of line 
            center_color: color of center
            distance_thresh: max distance
            line_thickness: line thickness
            center_thickness: center thickness
            half_line: draw half line or full line
        return: 
            image with line: np.ndarray
    """
    x, y = p[0], p[1]
    m_thresh = 100
    img = image

    _new_x, _new_y, new_x, new_y = get_dots(
        x, y, alpha,
        ignore_center=not half_line,
        distance_thresh=distance_thresh
    )

    if line_color is not None:
        img = cv.line(img, (int(_new_x), int(_new_y)), (int(
            new_x), int(new_y)), line_color, line_thickness)
    if center_color is not None:
        img = cv.circle(img, (int(x), int(y)), center_thickness,
                        center_color, center_thickness)
    return img


def std_draw_points(image: np.ndarray, point_0: POINT, point_1: POINT):
    radius = 3
    thickness = 3
    image = cv.circle(
        img=image,
        center=point_0,
        radius=radius,
        color=ACC['CENTER'],
        thickness=thickness)

    image = cv.circle(
        img=image,
        center=point_1,
        radius=radius,
        color=RAW['CENTER'],
        thickness=thickness)

    return image


def std_draw_line(image: np.ndarray, point: POINT, alpha: float, mode: DrawMode):
    if mode == DrawMode.Accept:
        line_color = ACC['LINE']
        center_color = ACC['CENTER']
    elif mode == DrawMode.Decline:
        line_color = DEC['LINE']
        center_color = DEC['CENTER']
    elif mode == DrawMode.Raw:
        line_color = RAW['LINE']
        center_color = RAW['CENTER']

    return draw_line(
        image=image,
        p=point,
        alpha=alpha,
        line_color=line_color,
        center_color=center_color,
        half_line=True,
        line_thickness=3
    )


def normalize_alpha(alpha):
    if alpha < 0:
        return alpha + 360
    if alpha > 360:
        return alpha - 360
    return alpha


def get_alpha(x0, y0, x1, y1):
    y = -1 * (y1 - y0)
    x = x1 - x0

    alpha = (np.arctan2(y, x) / math.pi) * 180
    return normalize_alpha(alpha)

class Visualizer:
    def __init__(self):
        pass

    def __call__(
        self,
        image: np.ndarray,
        image_name: int,
        accepted_predictions: list,
        declined_predictions: list,
        annotations: list,
        write_dir: str,
        predictions_store=None,
    ):
        path = osp.join(write_dir, f"{image_name:03d}.jpg")
        """
        for ann in annotations:
            x, y, alpha = int(ann[0]), int(ann[1]), int(ann[2])
            image = draw_line(image, (x, y), alpha, line_color=(0, 0, 0), center_color=(0, 0, 0), half_line=True, distance_thresh=40, line_thickness=2)
        for det in declined_predictions:
            x, y, alpha, score = det.astype(np.int64)
            image = draw_line(image, (x, y), alpha, line_color=(0, 0, 255), center_color=(0, 0, 0), half_line=True, distance_thresh=40, line_thickness=2)
        for det in accepted_predictions:
            x, y, alpha, score = det.astype(np.int64)
            image = draw_line(image, (x, y), alpha, line_color=(0, 255, 0), center_color=(0, 0, 0), half_line=True, distance_thresh=40, line_thickness=2)
        """
        if predictions_store is not None:
            image = image.astype(np.float64)
            image = np.zeros(shape=image.shape[:2], dtype=np.float64)
            scores = np.squeeze(predictions_store['classification'])
            regressions = np.squeeze(predictions_store['regression'])
            anchors = np.squeeze(predictions_store['anchors'])
            bg = scores <= 0.15
            fg = scores >= 0.85
            predictions = anchors.astype(np.int64)
            x = predictions[:, 0]
            y = predictions[:, 1]
            print(f"image: {image.shape} | x: {x.max()} & {x.min()} & {x.shape} | y: {y.max()} & {x.min()} & {x.shape}")
            print("anchors: {0}".format(anchors.max(axis=0)))
            y[y == image.shape[0]] = image.shape[0] - 1
            x[x == image.shape[1]] = image.shape[1] - 1
            yx = np.concatenate((y[:, np.newaxis], x[:, np.newaxis]), axis=1)
            print(yx.shape)
            bg_yx = yx[bg]
            print("bg_yx", bg_yx.shape, bg_yx.dtype)
            for position in bg_yx:
                try:
                    image[position[0] + 5, position[1]] = 255
                    # image[position[0], position[1]] *= 0.5
                except IndexError:
                    continue
            # image[bg_yx] *= 0.5
            # print("scores: {0} | {1}".format(scores.shape, scores.dtype))
            # print("regressions: {0} | {1}".format(regressions.shape, regressions.dtype))
            # print("anchors: {0} | {1}".format(anchors.shape, anchors.dtype))
            image = image.astype(np.uint8)
        cv.imwrite(path, image)
