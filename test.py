import os
import numpy as np
from cv2 import cv2
import torchvision
import csv
from os import path as osp
import torch
import shutil
from prediction import imageloader
from prediction.predict_boxes import split_uncertain_and_noisy
from retinanet.utils import load_classes
from utils.prediction import detect
from utils.visutils import draw_line
from retinanet import dataloader
from retinanet.utils import ActiveLabelMode
import retinanet


def pad_image(image):
    rows, cols = image.shape[0], image.shape[1]
    pad_w = 32 - (rows % 32)
    pad_h = 32 - (cols % 32)
    pad_l = pad_w // 2
    pad_r = (pad_w // 2) + (pad_w % 2)
    pad_t = pad_h // 2
    pad_d = (pad_h // 2) + (pad_h % 2)
    image = cv2.copyMakeBorder(image, pad_l, pad_r, pad_t, pad_d, cv2.BORDER_CONSTANT)
    return image


class Mask:
    def __init__(self, image):
        # self.canvas = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
        self.image = image.astype(np.float64)

    # def __call__(self, index, color):
    #     left, top, right, bottom = index
    #     self.canvas[top:bottom, left:right] = color

    def __call__(self, index, status):
        left, top, right, bottom = index
        if status:
            self.image[top:bottom, left:right] *= 0.5


class Dataset:
    def __init__(self, csv_annotations_path: str, labels_path: str, images_dir: str, extension: str = ".jpg"):
        self.class_to_index, self.index_to_class = load_classes(csv_class_list_path=labels_path)
        self.col_wise_annotations = dict()
        self.row_wise_annotations = dict()
        self.load_annotations(csv_annotations_path=csv_annotations_path)
        self.image_names = sorted(list(self.row_wise_annotations.keys()))
        self.images_dir = osp.expandvars(osp.expanduser(osp.abspath(images_dir)))
        self.ext = extension
        self._column_wise = False
        self._row_wise = not self._column_wise

    def load_annotations(self, csv_annotations_path: str):
        fileIO = open(csv_annotations_path, 'r')
        csv_reader = csv.reader(fileIO)
        for row in csv_reader:
            img_name, x, y, alpha, label, query_type = row
            x, y, alpha = int(round(float(x))), int(round(float(y))), int(round(float(alpha)))
            if query_type == 'asked':
                is_asked = True
                is_noisy = False
            elif query_type == 'noisy':
                is_asked = False
                is_noisy = True
            else:
                raise AssertionError("Invalid query type")
            label = int(self.class_to_index[label])
            self.fill_annotations(
                img_name=img_name, x=x, y=y, alpha=alpha, label=label, is_asked=is_asked, is_noisy=is_noisy,
            )

    def fill_annotations(self, img_name, x, y, alpha, label, is_asked, is_noisy):
        if not (img_name in self.col_wise_annotations.keys()):
            self.col_wise_annotations[img_name] = {
                'x': [], 'y': [], 'alpha': [], 'label': [], 'is_asked': [], 'is_noisy': [],
            }
        if not (img_name in self.row_wise_annotations.keys()):
            self.row_wise_annotations[img_name] = []
        self.col_wise_annotations[img_name]['x'].append(x)
        self.col_wise_annotations[img_name]['y'].append(y)
        self.col_wise_annotations[img_name]['alpha'].append(alpha)
        self.col_wise_annotations[img_name]['label'].append(label)
        self.col_wise_annotations[img_name]['is_asked'].append(is_asked)
        self.col_wise_annotations[img_name]['is_noisy'].append(is_noisy)
        self.row_wise_annotations[img_name].append(
            {'x': x, 'y': y, 'alpha': alpha, 'label': label, 'is_asked': is_asked, 'is_noisy': is_noisy, }
        )

    def load_image(self, index):
        image_path = osp.join(self.images_dir, self.image_names[index] + self.ext)
        image = cv2.imread(image_path)
        return image

    def __call__(self, image_name: (str, int)):
        image_index = self.image_names.index(image_name)
        if self._column_wise:
            annotations = self.col_wise_annotations[image_name]
        else:  # self._row_wise
            annotations = self.row_wise_annotations[image_name]
        image = self.load_image(index=image_index)
        data = {"img": image, "annots": annotations}
        return data

    def __getitem__(self, index):
        if self._column_wise:
            annotations = self.col_wise_annotations[self.image_names[index]]
        else:  # self._row_wise
            annotations = self.row_wise_annotations[self.image_names[index]]
        image = self.load_image(index=index)
        data = {"img": image, "annots": annotations}
        return data

    def set_column_wise(self, value: bool):
        if isinstance(value, bool):
            self._column_wise = value
            self._row_wise = not value

    def set_row_wise(self, value: bool):
        if isinstance(value, bool):
            self._row_wise = value
            self._column_wise = not value

    def is_column_wise(self):
        return self._column_wise

    def is_row_wise(self):
        return self._row_wise

    column_wise = property(is_column_wise, set_column_wise)
    row_wise = property(is_row_wise, set_row_wise)


class UncertaintyStatus:
    def __init__(self, loader, model, class_list_file_path, corrected_annotations_file_path, radius=50):
        self.NAME, self.X, self.Y, self.ALPHA, self.LABEL, self.TRUTH, self.SCORE = 0, 1, 2, 3, 5, 5, 4
        self.radius = radius
        self.loader = loader
        self.model = model
        self.class_to_index, self.index_to_class = load_classes(csv_class_list_path=class_list_file_path)
        self.corrected_annotations_file_path = corrected_annotations_file_path
        self.img_pattern = cv2.imread(osp.join(self.loader.img_dir, self.loader.image_names[0] + self.loader.ext))
        self.tile_states = np.full(shape=(len(self.loader.image_names), self.img_pattern.shape[0], self.img_pattern.shape[1]), fill_value=False, dtype=np.bool_)

    def _load_annotations(self, path: str) -> np.array:
        assert osp.isfile(path), "File does not exist."
        boxes = list()
        fileIO = open(path, "r")
        reader = csv.reader(fileIO, delimiter=",")
        for row in reader:
            if row[self.X] == row[self.Y] == row[self.ALPHA] == row[self.LABEL] == "":
                continue
            box = [None, None, None, None, None]
            box[self.NAME] = float(row[self.NAME])
            box[self.X] = float(row[self.X])
            box[self.Y] = float(row[self.Y])
            box[self.ALPHA] = float(row[self.ALPHA])
            box[self.LABEL] = float(self.class_to_index[row[self.LABEL]])
            boxes.append(box)
        fileIO.close()
        boxes = np.asarray(boxes, dtype=np.float64)
        return np.asarray(boxes[:, [self.NAME, self.X, self.Y, self.ALPHA, self.LABEL]], dtype=np.float64)

    def _get_previous_corrected_annotation_names(self):
        if osp.isfile(self.corrected_annotations_file_path):
            previous_corrected_annotations = self._load_annotations(path=self.corrected_annotations_file_path)
            previous_corrected_names = previous_corrected_annotations[:, self.NAME]
        else:
            previous_corrected_names = np.array(list(), dtype=np.float64)
        return previous_corrected_names

    def get_active_predictions(self):
        detections = detect(dataset=self.loader, retinanet_model=self.model)
        detections[:, [self.X, self.Y, self.ALPHA]] = np.around(detections[:, [self.X, self.Y, self.ALPHA]])
        previous_corrected_names = self._get_previous_corrected_annotation_names()
        print(f"detections: {detections.shape} {detections.dtype}\nprevious {len(previous_corrected_names)}")
        uncertain_boxes, noisy_boxes = split_uncertain_and_noisy(
            boxes=detections,
            previous_corrected_boxes_names=previous_corrected_names,
        )
        boxes = {"uncertain": uncertain_boxes, "noisy": noisy_boxes}
        return boxes

    def load_uncertainty_states(self, boxes):
        uncertain_boxes, noisy_boxes = boxes["uncertain"], boxes["noisy"]
        self.tile_states[:, :, :] = False
        r = self.radius
        for uncertain_box in uncertain_boxes:
            position = self.loader.image_names.index(f"{int(uncertain_box[self.NAME]):03d}")
            x = int(uncertain_box[self.X])
            y = int(uncertain_box[self.Y])
            self.tile_states[position, y-r:y+r, x-r:x+r] = True

    def get_mask(self, index):
        tile_mask = self.tile_states[index]
        return tile_mask


class ActiveStatus:
    def __init__(self, data_loader):
        self._data_loader = data_loader

    def _get_ground_truth_annotations(self, index):
        img = cv2.imread(self._data_loader.image_names[index])
        rows, cols = img.shape[0], img.shape[1]
        annotations = self._data_loader[index]["annot"].cpu().detach().numpy()
        is_valid = np.logical_and((annotations[:, 0] < cols), (annotations[:, 1] < rows))
        annotations = annotations[is_valid]
        return annotations

    @staticmethod
    def _correct_uncertains(annotations, uncertainty_state):
        X, Y = 0, 1
        gt_yx = annotations[:, Y].astype(np.int64), annotations[:, X].astype(np.int64)
        status = uncertainty_state[gt_yx]
        in_uncertain_gt_indices = np.squeeze(np.argwhere(status), axis=-1)
        in_uncertain_gt_annotations = annotations[in_uncertain_gt_indices]
        return in_uncertain_gt_annotations

    @staticmethod
    def concat_noisy_and_corrected_boxes(corrected_boxes, noisy_boxes):
        corrected_boxes = corrected_boxes[:, :5]
        corrected_col = np.full(shape=(corrected_boxes.shape[0], 1), fill_value=ActiveLabelMode.corrected.value, dtype=corrected_boxes.dtype)
        corrected_boxes = np.concatenate([corrected_boxes, corrected_col], axis=1)

        noisy_boxes = noisy_boxes[:, :5]
        noisy_col = np.full(shape=(noisy_boxes.shape[0], 1), fill_value=ActiveLabelMode.noisy.value, dtype=noisy_boxes.dtype)
        noisy_boxes = np.concatenate([noisy_boxes, noisy_col], axis=1)

        active_annotations = np.concatenate([corrected_boxes, noisy_boxes], axis=0)
        active_annotations = active_annotations[active_annotations[:, 0].argsort(), :]
        return active_annotations

    def correct(self, uncertainty_states):
        corrected_annotations = []
        for i in range(len(self._data_loader.image_names)):
            uncertainty_state = uncertainty_states[i]
            gt_annotations = self._get_ground_truth_annotations(index=i)
            if len(gt_annotations) == 0:
                continue
            in_uncertain_gt_annotations = ActiveStatus._correct_uncertains(
                annotations=gt_annotations,
                uncertainty_state=uncertainty_state,
            )
            img_number = int(osp.splitext(osp.basename(self._data_loader.image_names[i]))[0])
            img_number_col = np.full(shape=(in_uncertain_gt_annotations.shape[0], 1), fill_value=img_number)
            in_uncertain_gt_annotations = np.concatenate([img_number_col, in_uncertain_gt_annotations], axis=1)
            corrected_annotations.extend(in_uncertain_gt_annotations.tolist())
        corrected_annotations = np.asarray(corrected_annotations)
        return corrected_annotations


images_dir = osp.expanduser("~/Saffron/dataset/Train")
csv_classes = osp.abspath("annotations/labels.csv")
csv_anots = osp.abspath("annotations/test.csv")
dataset = Dataset(csv_annotations_path=csv_anots, labels_path=csv_classes, images_dir=images_dir)
dataset.column_wise = False

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
