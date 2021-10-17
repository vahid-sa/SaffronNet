import numpy as np
import csv
import glob
from cv2 import cv2
from os import path as osp

from retinanet.utils import ActiveLabelMode, load_classes, ActiveLabelModeSTR
from prediction.predict_boxes import split_uncertain_and_noisy
from utils.prediction import detect


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

    def write_states(self, directory):
        assert len(self.loader.image_names) == self.tile_states.shape[0]
        for i in range(len(self.loader.image_names)):
            state = self.tile_states[i]
            path = osp.join(directory, self.loader.image_names[i] + ".npy")
            with open(path, "wb") as fileIO:
                np.save(fileIO, state)


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


def write_corrected_annotations(annotations, path, class_list_path):
    IMG, X, Y, ALPHA, LABEL = 0, 1, 2, 3, 4
    fileIO = open(path, "w")
    writer = csv.writer(fileIO)
    _, index_to_class = load_classes(csv_class_list_path=class_list_path)
    for annotation in annotations:
        img_number, x, y, alpha, label = annotation[[IMG, X, Y, ALPHA, LABEL]].astype(np.int64)
        img_name = format(img_number, "03d")
        label_name = index_to_class[str(label)]
        row = [img_name, x, y, alpha, label_name]
        writer.writerow(row)
    fileIO.close()


def write_active_annotations(annotations, path, class_list_path):
    IMG, X, Y, ALPHA, LABEL, STATUS = 0, 1, 2, 3, 4, 5
    fileIO = open(path, "w")
    writer = csv.writer(fileIO)
    _, index_to_class = load_classes(csv_class_list_path=class_list_path)
    for annotation in annotations:
        img_number, x, y, alpha, label, status = annotation[[IMG, X, Y, ALPHA, LABEL, STATUS]].astype(np.int64)
        img_name = format(img_number, "03d")
        label_name = index_to_class[str(label)]
        if status == ActiveLabelMode.corrected.value:
            status_name = ActiveLabelModeSTR.gt.value
        elif status == ActiveLabelMode.noisy.value:
            status_name = ActiveLabelModeSTR.noisy.value
        else:
            raise ValueError(f"'{status} is not a correct value'")
        row = [img_name, x, y, 90 - alpha, label_name, status_name]
        writer.writerow(row)
    fileIO.close()
