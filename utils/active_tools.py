import os
import shutil
import logging
import numpy as np
import csv
import glob
from cv2 import cv2
from os import path as osp

from retinanet.utils import ActiveLabelMode, load_classes, ActiveLabelModeSTR
from utils.prediction import detect


class Active:
    def __init__(self, loader, states_dir, radius):
        self._NAME, self._X, self._Y, self._ALPHA, self._LABEL, self._SCORE = 0, 1, 2, 3, 4, 5
        self._loader = loader
        self._states_dir = states_dir
        self._states = self._init_states()
        self._radius = radius
        self._boxes = None
        self._uncertain_indices = None
        self._noisy_indices = None
        self._gt_annotations = None
        self._active_annotations = None

        if osp.isdir(self._states_dir):
            shutil.rmtree(self._states_dir)
        os.makedirs(self._states_dir, exist_ok=False)

    def _init_states(self):
        img_pattern = cv2.imread(osp.join(self._loader.img_dir, self._loader.image_names[0] + self._loader.ext))
        states = np.full(
            shape=(len(self._loader.image_names), img_pattern.shape[0], img_pattern.shape[1]),
            fill_value=False,
            dtype=np.bool_,
        )
        return states

    def _load_previous_states(self):
        paths = glob.glob(osp.join(self._states_dir, "*.npy"))
        for path in paths:
            f = open(path, "rb")
            state = np.load(f)
            f.close()
            name = osp.splitext(osp.basename(path))[0]
            index = self._loader.image_names.index(name)
            self._states[index, :, :] = state

    def _predict_boxes(self, model):
        self._boxes = detect(dataset=self._loader, retinanet_model=model)
        self._boxes[:, [self._X, self._Y, self._ALPHA]] = np.around(self._boxes[:, [self._X, self._Y, self._ALPHA]])

    def _select_uncertain_boxes(self, budget):
        indices = list()
        scores = np.abs(self._boxes[:, self._SCORE] - 0.5)
        sorted_indices = scores.argsort()
        sorted_boxes = self._boxes[sorted_indices]
        i = 0
        while len(indices) < budget:
            if i >= len(sorted_boxes):
                break
            box = sorted_boxes[i]
            position = self._loader.image_names.index(f"{int(box[self._NAME]):03d}")
            x, y = int(box[self._X]), int(box[self._Y])
            if not self._states[position, y, x]:
                indices.append(sorted_indices[i])
            i += 1
        self._uncertain_indices = indices

    def _select_ground_truth_states(self):
        for box in self._boxes[self._uncertain_indices]:
            name, x, y = int(box[self._NAME]), int(box[self._X]), int(box[self._Y])
            position = int(self._loader.image_names.index(f"{name:03d}"))
            r = self._radius
            self._states[position, y-r:y+r, x-r:x+r] = True

    @staticmethod
    def _get_ground_truth_annotations(dataloader, index):
        img = cv2.imread(dataloader.image_names[index])
        rows, cols = img.shape[0], img.shape[1]
        annotations = dataloader[index]["annot"].cpu().detach().numpy()
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

    def _correct(self, dataloader):
        corrected_annotations = []
        for i in range(len(dataloader.image_names)):
            uncertainty_state = self._states[i]
            gt_annotations = Active._get_ground_truth_annotations(dataloader=dataloader, index=i)
            if len(gt_annotations) == 0:
                continue
            in_uncertain_gt_annotations = Active._correct_uncertains(
                annotations=gt_annotations,
                uncertainty_state=uncertainty_state,
            )
            img_number = int(osp.splitext(osp.basename(dataloader.image_names[i]))[0])
            img_number_col = np.full(shape=(in_uncertain_gt_annotations.shape[0], 1), fill_value=img_number)
            in_uncertain_gt_annotations = np.concatenate([img_number_col, in_uncertain_gt_annotations], axis=1)
            corrected_annotations.extend(in_uncertain_gt_annotations.tolist())
        corrected_annotations = np.asarray(corrected_annotations)
        self._gt_annotations = corrected_annotations

    def _select_noisy_boxes(self):
        indices = list()
        uncertain_boxes = self._boxes[self._uncertain_indices]
        uncertain_scores = uncertain_boxes[:, self._SCORE]
        maximum_uncertain_score = np.max(uncertain_scores)
        higher_than_uncertain_score_indices = np.squeeze(np.argwhere(self._boxes[:, self._SCORE] > maximum_uncertain_score))
        for i in higher_than_uncertain_score_indices:
            box = self._boxes[i]
            name, x, y = int(box[self._NAME]), int(box[self._X]), int(box[self._Y])
            position = int(self._loader.image_names.index(f"{name:03d}"))
            if (not self._states[position, y, x]) and (True in self._states[position, :, :]):
                indices.append(i)
        self._noisy_indices = indices

    def _concat_noisy_and_corrected_boxes(self):
        noisy_boxes = self._boxes[self._noisy_indices]
        corrected_boxes = self._gt_annotations[:, :5]
        corrected_col = np.full(shape=(corrected_boxes.shape[0], 1), fill_value=ActiveLabelMode.corrected.value, dtype=corrected_boxes.dtype)
        corrected_boxes = np.concatenate([corrected_boxes, corrected_col], axis=1)

        noisy_boxes = noisy_boxes[:, :5]
        noisy_col = np.full(shape=(noisy_boxes.shape[0], 1), fill_value=ActiveLabelMode.noisy.value, dtype=noisy_boxes.dtype)
        noisy_boxes = np.concatenate([noisy_boxes, noisy_col], axis=1)

        active_annotations = np.concatenate([corrected_boxes, noisy_boxes], axis=0)
        active_annotations = active_annotations[active_annotations[:, 0].argsort(), :]
        self._active_annotations = active_annotations

    def _write_states(self):
        if osp.isdir(self._states_dir):
            shutil.rmtree(self._states_dir)
        os.makedirs(self._states_dir, exist_ok=False)
        for i in range(self._states.shape[0]):
            states = self._states[i]
            name = self._loader.image_names[i] + ".npy"
            path = osp.join(self._states_dir, name)
            with open(path, "wb") as f:
                np.save(file=f, arr=states)

    def _write_ground_truth_annotations(self, ground_truth_path, class_list_path):
        IMG, X, Y, ALPHA, LABEL = 0, 1, 2, 3, 4
        os.makedirs(osp.dirname(ground_truth_path), exist_ok=True)
        f = open(ground_truth_path, 'w')
        csv_writer = csv.writer(f, delimiter=',')
        _, index_to_class = load_classes(csv_class_list_path=class_list_path)
        for annotation in self._gt_annotations:
            img_number, x, y, alpha, label = annotation[[IMG, X, Y, ALPHA, LABEL]].astype(np.int64)
            img_name = format(img_number, "03d")
            label_name = index_to_class[str(label)]
            row = [img_name, x, y, alpha, label_name]
            csv_writer.writerow(row)
        f.close()

    def _write_active_annotations(self, path, class_list_path):
        IMG, X, Y, ALPHA, LABEL, STATUS = 0, 1, 2, 3, 4, 5
        os.makedirs(osp.dirname(path), exist_ok=True)
        fileIO = open(path, "w")
        writer = csv.writer(fileIO)
        _, index_to_class = load_classes(csv_class_list_path=class_list_path)
        for annotation in self._active_annotations:
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

    def create_active_annotations(
            self,
            model,
            budget,
            ground_truth_loader,
            ground_truth_annotations_path,
            active_annotations_path,
            classes_list_path,
    ):
        self._boxes = None
        self._uncertain_indices = None
        self._noisy_indices = None
        self._gt_annotations = None
        self._active_annotations = None
        self._states = self._init_states()

        logging.info("Loading previos states...")
        self._load_previous_states()
        logging.info("Predicting boxes...")
        self._predict_boxes(model=model)
        logging.info("Selecting uncertain boxes...")
        self._select_uncertain_boxes(budget=budget)
        logging.info("Selecting ground truth states...")
        self._select_ground_truth_states()
        logging.info("Correcting uncertain boxes...")
        self._correct(dataloader=ground_truth_loader)
        logging.info("Selecting noisy boxes...")
        self._select_noisy_boxes()
        logging.info("Creating active annotations...")
        self._concat_noisy_and_corrected_boxes()
        logging.info("Writing...")
        self._write_states()
        self._write_ground_truth_annotations(
            ground_truth_path=ground_truth_annotations_path,
            class_list_path=classes_list_path,
        )
        self._write_active_annotations(
            path=active_annotations_path,
            class_list_path=classes_list_path,
        )
        logging.info("Creating active annotations Done!")

    @property
    def states(self):
        return self._states

    @property
    def predictions(self):
        return self._boxes

    @property
    def uncertain_predictions(self):
        if (self._boxes is None) or (self._uncertain_indices is None):
            value = None
        else:
            value = self._boxes[self._uncertain_indices]
        return value

    @property
    def noisy_predictions(self):
        if (self._boxes is None) or (self._noisy_indices is None):
            value = None
        else:
            value = self._boxes[self._noisy_indices]
        return value

    @property
    def active_annotations(self):
        return self._active_annotations

    @property
    def ground_truth_annotations(self):
        return self._gt_annotations
