import os
import cv2
import math
import time
import csv
import numpy as np
from copy import deepcopy
from os import path as osp
from threading import Thread
from utils.annotation_tools import Annotation

def write_annotations(annotations: list, path: str):
    f = open(path, 'w')
    writer = csv.writer(f)
    writer.writerows(annotations)
    f.close()

def write_vid_frame():
    t = time.time()
    while not end:
        if (time.time() - t) >= (1.0 / fps):
            out_vid.write(annotator.image)
            t = time.time()
    return 0

images_directory = osp.expanduser("~/Documents/thesis/Saffron/annotation")
image_paths = []
for root, dirs, files in os.walk(images_directory):
    for file in files:
        image_paths.append(osp.join(root, file))


fps = 30
out_vid = cv2.VideoWriter(
    osp.expanduser("~/Documents/thesis/supervised_annotating.mp4"),
    cv2.VideoWriter_fourcc("m", "p", "4", "v"),
    fps,
    (1296, 972),
)
start_time = time.time()
num_annotations = 0
cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
break_process = False
for i, image_path in enumerate(image_paths):
    root, file = osp.split(image_path)
    name, ext = osp.splitext(file)
    annotation_path = osp.join(osp.dirname(root), name + '.csv')
    image = cv2.imread(image_path)
    annotator = Annotation(image=image)
    cv2.setMouseCallback('image', annotator.annotate)
    annotatons_length = 0
    end = False
    t = Thread(target=write_vid_frame, args=())
    t.start()
    while True:
        img = annotator.image
        cv2.putText(
            img,
            f"{int(time.time() - start_time)}||{num_annotations}",
            (50, 50),
            0,
            5e-3 * 150,
            (255, 255, 0),
            2,
        )
        cv2.imshow('image', img)
        if len(annotator.annotations) > annotatons_length:
            num_annotations += (len(annotator.annotations) - annotatons_length)
            annotatons_length = len(annotator.annotations)
        key = cv2.waitKey(20)
        if key == 13:
            write_annotations(annotations=annotator.annotations, path=annotation_path)
            break
        elif key == 27:
            break_process = True
            break
    end = True
    if break_process:
        break
cv2.destroyAllWindows()
out_vid.release()
