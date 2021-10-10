import numpy as np
from cv2 import cv2
import torchvision
from os import path as osp
import torch
from prediction import imageloader
from utils.visutils import draw_line
from retinanet import dataloader
from retinanet.utils import ActiveLabelMode
import retinanet

from utils.active_tools import ActiveStatus, UncertaintyStatus, write_corrected_annotations, write_active_annotations, write_states


images_dir = osp.expanduser("~/Saffron/dataset/Train")
csv_classes = osp.abspath("annotations/labels.csv")
csv_anots = osp.abspath("annotations/test.csv")

image_loader = imageloader.CSVDataset(
    filenames_path="annotations/filenames.json",
    partition="unsupervised",
    class_list=csv_classes,
    images_dir=images_dir,
    image_extension=".jpg",
    transform=torchvision.transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
)

data_loader = dataloader.CSVDataset(
    train_file="annotations/unsupervised.csv",
    class_list=csv_classes,
    images_dir=images_dir,
    transform=torchvision.transforms.Compose([dataloader.Normalizer(), dataloader.Resizer()]),
)

retinanet_model = torch.load(osp.expanduser('~/Saffron/weights/supervised/init_model.pt'))
retinanet.settings.NUM_QUERIES = 100
retinanet.settings.NOISY_THRESH = 0.15

uncertainty_status = UncertaintyStatus(
    loader=image_loader,
    model=retinanet_model,
    class_list_file_path="annotations/labels.csv",
    corrected_annotations_file_path="active_annotations/corrected.csv",
)
images_detections = uncertainty_status.get_active_predictions()
uncertainty_status.load_uncertainty_states(boxes=images_detections)

uncertain_detections = images_detections["uncertain"]
noisy_detections = images_detections["noisy"]

active_status = ActiveStatus(data_loader=data_loader)
corrected_annotations = active_status.correct(uncertainty_states=uncertainty_status.tile_states)
active_annotations = ActiveStatus.concat_noisy_and_corrected_boxes(corrected_boxes=corrected_annotations, noisy_boxes=noisy_detections)
active_states = uncertainty_status.tile_states

direc = osp.expanduser('~/tmp/saffron_imgs')
for i in range(len(image_loader.image_names)):
    image = cv2.imread(osp.join(image_loader.img_dir, image_loader.image_names[i] + image_loader.ext))
    my_mask = np.ones(shape=image.shape, dtype=np.float64)
    mask = active_states[i]
    my_mask[mask] *= 0.5
    image = image.astype(np.float64) * my_mask
    image = image.astype(np.uint8)
    image_noisy_detections = noisy_detections[noisy_detections[:, 0] == int(image_loader.image_names[i])]
    image_uncertain_detections = uncertain_detections[uncertain_detections[:, 0] == int(image_loader.image_names[i])]
    image_active_annotations = active_annotations[active_annotations[:, 0] == int(image_loader.image_names[i])]
    image_corrected_annotations = corrected_annotations[corrected_annotations[:, 0] == int(image_loader.image_names[i])]

    for det in image_uncertain_detections:
        x = int(det[1])
        y = int(det[2])
        alpha = det[3]
        score = det[5]
        image = draw_line(image, (x, y), alpha, line_color=(0, 0, 255), center_color=(0, 0, 0), half_line=True, distance_thresh=40, line_thickness=2)
        cv2.putText(image, str(round(score, 2)), (x + 3, y + 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    for det in image_noisy_detections:
        x = int(det[1])
        y = int(det[2])
        alpha = det[3]
        score = det[5]
        image = draw_line(image, (x, y), alpha, line_color=(0, 255, 255), center_color=(0, 0, 0), half_line=True,
                          distance_thresh=40, line_thickness=2)
        cv2.putText(image, str(round(score, 2)), (x + 3, y + 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)

    for annot in image_corrected_annotations:
        x = int(annot[1])
        y = int(annot[2])
        alpha = annot[3]
        score = annot[5]
        image = draw_line(image, (x, y), alpha, line_color=(255, 255, 0), center_color=(0, 0, 0), half_line=True,
                          distance_thresh=40, line_thickness=2)

    for annot in image_active_annotations:
        x = int(annot[1])
        y = int(annot[2])
        alpha = annot[3]
        status = annot[-1]
        if status == ActiveLabelMode.corrected.value:
            color = (0, 255, 0)
        elif status == ActiveLabelMode.noisy.value:
            color = (255, 0, 0)
        else:
            color = (0, 0, 0)
        image = draw_line(image, (x, y), alpha, line_color=color, center_color=(0, 0, 0), half_line=True,
                          distance_thresh=40, line_thickness=2)
        cv2.imwrite(osp.join(direc, image_loader.image_names[i] + image_loader.ext), image)

corrected_annotations_path = osp.expanduser("~/Saffron/active_annotations/corrected.csv")
active_annotations_path = osp.expanduser("~/Saffron/active_annotations/train.csv")
active_states_path = osp.expanduser("~/Saffron/active_annotations/states.npy")
write_corrected_annotations(annotations=corrected_annotations, path=corrected_annotations_path, class_list_path=csv_classes)
write_active_annotations(annotations=active_annotations, path=active_annotations_path, class_list_path=csv_classes)
write_states(states=active_states, path=active_states_path)
