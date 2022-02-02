import os
import cv2
from cv2 import FlannBasedMatcher
import numpy as np
from os import path as osp
from utils.annotation_tools import Annotation, retrieve_patches


states_dir = osp.expanduser("~/Documents/thesis/Saffron/states/")
images_dir = osp.expanduser("~/Documents/thesis/Saffron/dataset/Train/")

sample_numbers = [int(osp.splitext(name)[0]) for name in os.listdir(states_dir)]

cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
break_process = False
for sample_number in sample_numbers:
    if break_process:
        break
    string_number = f"{sample_number:03d}"
    state_path = osp.join(states_dir, string_number + ".npy")
    image_path = osp.join(images_dir, string_number + ".jpg")
    image = cv2.imread(image_path)
    f = open(state_path, "rb")
    state = np.load(f)
    f.close()
    image = image.astype(np.float64)
    image[np.logical_not(state)] /= 2.0
    image = image.astype(np.uint8)
    patches = retrieve_patches(state=state)
    image_annotations = list()
    for patch in patches:
        if break_process:
            break
        l, t, r, b = patch
        margin = 0
        img = image[l - margin: r + margin, t - margin: b + margin]
        cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 3)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        continue
        annotator = Annotation(image=img)
        cv2.setMouseCallback('image', annotator.annotate)
        while break_process:
            img = annotator.image
            cv2.imshow('image', img)
            if cv2.waitKey(20) & 0xFF == 27:
                break_process = True
        if break_process:
            break
        key = cv2.waitKey(0)
        if key == 13:
            continue
        elif key == 27:
            break_process = True
        else:
            continue
cv2.destroyAllWindows()
