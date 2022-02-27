import logging
import math
import numpy as np
import csv
from os import path as osp

from retinanet.model import VGGNet
from retinanet.utils import load_classes
from utils.prediction import detect
from retinanet import dataloader
from prediction import imageloader
from utils.visutils import normalize_alpha


class Active:
    def __init__(
        self,
        loader: imageloader.CSVDataset,
        annotations_path: str,
        class_list_path: str,
        ground_truth_dataloader: dataloader.CSVDataset,
        budget: int = 2,
        aggregator_type: str = "max",
        uncertainty_algorithm: str = "least",
        is_strategic: bool = True,
    ):
        self._NAME, self._X, self._Y, self._ALPHA, self._LABEL, self._SCORE = 0, 1, 2, 3, 4, 5
        self._loader = loader
        self._aggregator_type = aggregator_type
        self._uncertainty_algorithm = uncertainty_algorithm
        self._budget = budget
        self._annotations_path = annotations_path
        self._class_list_path = class_list_path
        self._gt_loader: dataloader.CSVDataset = ground_truth_dataloader

        self._boxes: np.ndarray = None
        self._uncertain_image_list: np.ndarray = None
        self._is_strategic = is_strategic
        if self._is_strategic:
            self._aggregator_type = "max"
            self._uncertainty_algorithm = "least"

    def _load_annotations(self) -> np.ndarray:
        class_to_index, _ = load_classes(csv_class_list_path=self._class_list_path)
        fileIO = open(self._annotations_path, "r")
        annotations = list()
        for row in csv.reader(fileIO):
            annotation = [None] * 5
            annotation[self._NAME] = int(row[self._NAME])
            annotation[self._X] = int(float(row[self._X]))
            annotation[self._Y] = int(float(row[self._Y]))
            annotation[self._ALPHA] = 90 - int(float(row[self._ALPHA]))
            annotation[self._LABEL] = class_to_index[row[self._LABEL]]
            annotations.append(annotation)
        fileIO.close()
        annotations = np.asarray(annotations, dtype=np.float32)
        return annotations

    def _predict_boxes(self, model: VGGNet) -> np.ndarray:
        boxes = detect(dataset=self._loader, retinanet_model=model)
        boxes[:, [self._X, self._Y, self._ALPHA]] = np.around(boxes[:, [self._X, self._Y, self._ALPHA]])
        return boxes

    @staticmethod
    def _calculate_least_confidence_scores(classification_scores: np.ndarray) -> np.ndarray:
        """calculates least confidence score for each element(box)

        Args:
            classification_scores (np.ndarray): (N,) 0 < score < 1

        Returns:
            np.ndarray: (N,) 0 < least confidence scores < 1
        """
        distance = np.abs(np.subtract(classification_scores, 0.5))
        least_confidence_scores = np.subtract(1.0, distance)
        return least_confidence_scores

    @staticmethod
    def _calculate_binary_cross_entropy_scores(classification_scores: np.ndarray) -> np.ndarray:
        """calculates least confidence score for each element(box)

        Args:
            classification_scores (np.ndarray): (N,) 0 < score < 1

        Returns:
            np.ndarray: (N,) 0 < binary cross entropy scores < 1
        """
        complement = np.subtract(1.0, classification_scores)
        entropy = np.multiply(classification_scores, np.log(classification_scores))
        complement_entropy = np.multiply(complement, np.log(complement))
        neg_bce = np.add(entropy, complement_entropy)
        bce = np.negative(neg_bce)
        return bce

    @staticmethod
    def _calculate_random_scores(classification_scores: np.ndarray) -> np.ndarray:
        """samples a random score for each element(box)

        Args:
            classification_scores (np.ndarray): (N,)

        Returns:
            np.ndarray: (N,) 0 < random value < 1
        """
        scores = np.random.uniform(size=classification_scores.shape)
        return scores

    def _calculate_image_uncertainty_score(self, predicted_boxes):
        if self._aggregator_type == "max":
            aggregator = np.max
        elif self._aggregator_type == "avg":
            aggregator = np.mean
        elif self._aggregator_type == "sum":
            aggregator = np.sum
        else:
            raise AssertionError("Incorrect input argument 'type'")
        if self._uncertainty_algorithm == "least":
            calculator = Active._calculate_least_confidence_scores
        elif self._uncertainty_algorithm == "bce":
            calculator = Active._calculate_binary_cross_entropy_scores
        elif self._uncertainty_algorithm == "random":
            calculator = Active._calculate_random_scores
            aggregator = np.mean
        else:
            raise AssertionError("Incorrect input argument 'algorithm'")
        uncertainty_scores = calculator(predicted_boxes[:, self._SCORE])
        boxes_names = predicted_boxes[:, self._NAME]
        img_names = np.unique(boxes_names)
        gt_loader_img_names = [int(osp.splitext(osp.basename(image_name))[0]) for image_name in self._gt_loader.image_names]
        image_uncertainty_scores = [
            [
                img_name,
                aggregator(uncertainty_scores[boxes_names == img_name]),
                len(self._gt_loader[gt_loader_img_names.index(img_name)]['annot']),
            ] for img_name in img_names
        ]
        return np.asarray(image_uncertainty_scores, dtype=np.float64)

    def _select_uncertain_images(self, annotated_img_names: list, img_uncertainty_scores: np.ndarray) -> list:
        not_selected_before = np.in1d(img_uncertainty_scores[:, 0], annotated_img_names, invert=True)
        not_selected_img_uncertainty_scores = img_uncertainty_scores[not_selected_before]
        if len(not_selected_img_uncertainty_scores) == 0:
            uncertain_img_names =  []
        else:
            if self._is_strategic:
                us = not_selected_img_uncertainty_scores[:, 1]
                sqrt_scores = np.sqrt(us)
                scores = np.exp(sqrt_scores) / np.sum(np.exp(sqrt_scores))
                indices_ = np.random.choice(
                    len(scores), scores.shape, replace=False, p=scores)
                sorted_img_uncertainty_scores = not_selected_img_uncertainty_scores[indices_]
            else:
                sorted_img_uncertainty_scores = not_selected_img_uncertainty_scores[np.argsort(not_selected_img_uncertainty_scores[:, 1])[::-1]]
            num_annotations = 0
            diff_annots_budget = self._budget
            index = -1
            for i in range(len(sorted_img_uncertainty_scores)):
                if num_annotations >= self._budget:
                    break
                num_new_annotations = num_annotations + sorted_img_uncertainty_scores[i][2]
                diff_new_annots_budget = np.abs(self._budget - num_new_annotations)
                if (diff_new_annots_budget < diff_annots_budget) or (i == 0):
                    num_annotations = num_new_annotations
                    diff_annots_budget = diff_new_annots_budget
                else:
                    break
                index = i
            uncertain_img_names = sorted_img_uncertainty_scores[:index + 1, 0].tolist()
        return uncertain_img_names

    def _label_uncertain_images(self, uncertain_img_names: list) -> np.ndarray:
        loader_image_paths = self._gt_loader.image_names
        loader_image_names = [int(osp.basename(osp.splitext(path)[0])) for path in loader_image_paths]
        annotations = []
        for img_name in uncertain_img_names:
            index =  loader_image_names.index(img_name)
            image_annotations = self._gt_loader[index]["annot"].numpy()
            image_annotations = np.hstack([
                np.full(shape=(image_annotations.shape[0], 1), dtype=image_annotations.dtype, fill_value=img_name),
                image_annotations,
                ])
            annotations.extend(image_annotations.tolist())
        return np.asarray(annotations, dtype=np.float32)

    def _write_annotations(self, annotations: np.ndarray):
        _, index_to_class = load_classes(csv_class_list_path=self._class_list_path)
        fileIO = open(self._annotations_path, "w")
        writer = csv.writer(fileIO)
        annots = annotations
        for annot in annots:
            row = [None] * 5
            row[self._NAME] = f"{int(annot[self._NAME]):03d}"
            row[self._X] = str(float(annot[self._X]))
            row[self._Y] = str(float(annot[self._Y]))
            # row[self._ALPHA] = str(float(normalize_alpha(90 - annot[self._ALPHA])))
            row[self._ALPHA] = str(float(90 - annot[self._ALPHA]))
            row[self._LABEL] = index_to_class[str(int(annot[self._LABEL]))]
            writer.writerow(row)
        fileIO.close()

    def create_annotations(
        self,
        model: VGGNet,
    ):
        predicted_boxes = self._predict_boxes(model=model)
        self._boxes = predicted_boxes
        img_uncertainty_scores = self._calculate_image_uncertainty_score(predicted_boxes=predicted_boxes)
        if osp.isfile(self._annotations_path):
            previous_annotations = self._load_annotations()
            annotated_img_names = previous_annotations[:, 0]
        else:
            previous_annotations = np.asarray([], dtype=np.float32)
            annotated_img_names = np.asarray([], dtype=np.float32)
        uncertain_img_names = self._select_uncertain_images(annotated_img_names=annotated_img_names, img_uncertainty_scores=img_uncertainty_scores)
        new_annotations = self._label_uncertain_images(uncertain_img_names=uncertain_img_names)
        self._uncertain_image_list = np.unique(new_annotations[:, 0].astype(np.int64)).tolist() if len(new_annotations) else []
        if len(previous_annotations) and len(new_annotations):
            annotations = np.concatenate([previous_annotations, new_annotations], axis=0)
        elif len(new_annotations):
            annotations = new_annotations
        elif len(previous_annotations):
            annotations = previous_annotations
        else:
            annotations = np.asarray([], dtype=np.float64)
        self._write_annotations(annotations=annotations)

    @property
    def predictions(self) -> np.ndarray:
        return self._boxes

    @property
    def uncertain_images(self) -> list:
        return self._uncertain_image_list
