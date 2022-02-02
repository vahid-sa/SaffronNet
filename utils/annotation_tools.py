import cv2
import math
import numpy as np
from copy import deepcopy


class Annotation:
    def __init__(self, image: np.ndarray):
        self._img: np.ndarray = image.copy()
        self._original_img: np.ndarray = image.copy()
        self._temp_img: np.ndarray = np.array([], dtype=np.uint8)
        self._start_point: tuple = tuple()
        self._mid_point: tuple = tuple()
        self._end_point: tuple = tuple()
        self._img_queue: np.ndarray = np.expand_dims(image.copy(), axis=0)
        self._annotations = list() # x, y, angle

    def _push(self):
        self._img_queue = np.concatenate([self._img_queue, np.expand_dims(self._img.copy(), axis=0)], axis=0)
        annotation = self._get_annotation_parameters()
        self._annotations.append(annotation)

    def _pop(self):
        if self._annotations:
            self._img_queue = np.delete(arr=self._img_queue, obj=-1, axis=0)
            _ = self._annotations.pop()
        self._img = self._img_queue[-1].copy()
    
    def _reset(self):
        self._img_queue = np.expand_dims(self._img_queue[0], axis=0)
        self._annotations = list()
        self._img = self._original_img.copy()

    def annotate(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._start_point = x, y
            cv2.circle(self._img, self._start_point, 5, (0, 0, 0), -1)
            self._temp_img = self._img.copy()
        elif event == cv2.EVENT_MOUSEMOVE and flags in (1, 9, 33):
            self._end_point = x, y
            self._img = self._temp_img.copy()
            cv2.line(self._img, self._start_point, self._end_point, (0, 0, 255), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self._img = self._temp_img
            self._end_point = x, y
            cv2.line(self._img, self._start_point, self._end_point, (0, 255, 0), 2)
            cv2.circle(self._img, self._start_point, 5, (0, 0, 0), -1)
            self._push()
        elif event == cv2.EVENT_RBUTTONUP  and flags == cv2.EVENT_FLAG_RBUTTON:
            self._pop()
        elif event == cv2.EVENT_RBUTTONDBLCLK and flags == 58:  #  rbuttondbclk + ctrl + alt + shift
            self._reset()

    def _get_annotation_parameters(self) -> float:
        x, y = self._start_point
        x2, y2 = self._end_point
        dx = x2 - x
        dy = y2 - y
        angle_radian = math.atan2(dy, dx)
        angle_degree = math.degrees(angle_radian)
        angle = int(round(angle_degree))
        annotation = [x, y, angle]
        return annotation
    
    @property
    def image(self):
        return self._img.copy()
    
    @property
    def annotations(self):
        return deepcopy(self._annotations)


def get_patch_params(contour: np.ndarray):
    cnt = np.squeeze(contour)
    left = np.min(cnt[:, 0])
    right = np.max(cnt[:, 0])
    top = np.min(cnt[:, 1])
    down = np.max(cnt[:, 1])
    patch = [left, top, right, down]
    return patch


def retrieve_patches(state: np.ndarray) -> list:
    im = np.zeros(shape=state.shape, dtype=np.uint8)
    im[state] = 255
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    patches = [get_patch_params(contour) for contour in contours]
    return patches
