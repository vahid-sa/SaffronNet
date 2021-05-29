from argparse import RawDescriptionHelpFormatter
import math
from enum import Enum
import cv2 as cv
import numpy as np
from typing import Tuple
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
