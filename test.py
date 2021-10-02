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
    def __init__(self, loader, model, class_list_file_path, corrected_annotations_file_path, tiling_size=100):
        self.NAME, self.X, self.Y, self.ALPHA, self.LABEL, self.TRUTH, self.SCORE = 0, 1, 2, 3, 4, 5, 5
        self.loader = loader
        self.model = model
        self.class_to_index, self.index_to_class = load_classes(csv_class_list_path=class_list_file_path)
        self.corrected_annotations_file_path = corrected_annotations_file_path
        self.img_pattern = cv2.imread(osp.join(self.loader.img_dir, self.loader.image_names[0] + self.loader.ext))
        self.tiling_indices = UncertaintyStatus._tile(image=self.img_pattern, size=tiling_size)
        self.tiling_states = np.full(
            shape=(len(self.loader.image_names), len(self.tiling_indices)),
            fill_value=False,
            dtype=np.bool,
        )

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

    @staticmethod
    def _tile(image, size):
        top, left = np.mgrid[0:image.shape[0]:size, 0:image.shape[1]:size]
        right = left + size - 1
        bottom = top + size - 1
        right[:, -1] = image.shape[1] - 1
        bottom[:, -1] = image.shape[0] - 1
        left = np.expand_dims(left, axis=2)
        top = np.expand_dims(top, axis=2)
        right = np.expand_dims(right, axis=2)
        bottom = np.expand_dims(bottom, axis=2)
        indices = np.concatenate([left, top, right, bottom], axis=-1)
        indices = indices.reshape((-1, indices.shape[-1]))
        return indices

    def get_active_predictions(self):
        detections = detect(dataset=self.loader, retinanet_model=self.model)
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
        self.tiling_states[:, :] = False
        for uncertain_box in uncertain_boxes:
            position = self.loader.image_names.index(f"{int(uncertain_box[self.NAME]):03d}")
            x, y = uncertain_box[[self.X, self.Y]]
            state = list(
                np.logical_and(
                    np.logical_and(
                        np.logical_and(self.tiling_indices[:, 0] <= x, self.tiling_indices[:, 1] <= y),
                        self.tiling_indices[:, 2] >= x),
                    self.tiling_indices[:, 3] >= y,
                )
            )
            # Remove noisy boxes in uncertain states
            if True in state:
                self.tiling_states[position, state.index(True)] = True
        return self.tiling_states

    def get_mask(self, index):
        img = cv2.imread(osp.join(self.loader.img_dir, self.loader.image_names[index] + self.loader.ext))
        mask = np.full(shape=img.shape, fill_value=False, dtype=np.bool)
        img_states = self.tiling_states[index]
        if True in img_states:
            for index, state in zip(self.tiling_indices, img_states):
                left, top, right, bottom = index
                if state:
                    mask[top:bottom, left:right] = True
        return mask


images_dir = osp.expanduser("~/Saffron/dataset/Train")
csv_classes = osp.abspath("annotations/labels.csv")
csv_anots = osp.abspath("annotations/test.csv")
dataset = Dataset(csv_annotations_path=csv_anots, labels_path=csv_classes, images_dir=images_dir)
dataset.column_wise = False


data_loader = imageloader.CSVDataset(
    filenames_path="annotations/filenames.json",
    partition="unsupervised",
    class_list=csv_classes,
    images_dir=images_dir,
    image_extension=".jpg",
    transform=torchvision.transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
)

retinanet_model = torch.load(osp.expanduser('~/Saffron/weights/supervised/init_model.pt'))
retinanet.settings.NUM_QUERIES = 100
retinanet.settings.NOISY_THRESH = 0.15

uncertainty_status = UncertaintyStatus(
    loader=data_loader,
    model=retinanet_model,
    class_list_file_path="annotations/labels.csv",
    corrected_annotations_file_path="active_annotations/corrected.csv",
)
images_detections = uncertainty_status.get_active_predictions()
images_states = uncertainty_status.load_uncertainty_states(boxes=images_detections)
uncertain_detections = images_detections["uncertain"]
noisy_detections = images_detections["noisy"]
print("noisy_detections", noisy_detections.shape)
direc = osp.expanduser('~/tmp/saffron_imgs')
if osp.isdir(direc):
    shutil.rmtree(direc)
os.makedirs(direc, exist_ok=False)
for i in range(len(data_loader.image_names)):
    mask = uncertainty_status.get_mask(index=i)
    # if not (True in mask):
    #     continue
    image = cv2.imread(osp.join(data_loader.img_dir, data_loader.image_names[i] + data_loader.ext)).astype(np.float64)
    my_mask = np.ones(shape=mask.shape, dtype=np.float64)
    my_mask[mask] *= 0.5
    image *= my_mask
    image = image.astype(np.uint8)
    image_uncertain_detections = uncertain_detections[uncertain_detections[:, 0] == int(data_loader.image_names[i])]
    image_noisy_detections = noisy_detections[noisy_detections[:, 0] == int(data_loader.image_names[i])]
    if image_noisy_detections.shape[0] == 0 and image_uncertain_detections.shape[0] == 0:
        continue
    for det in image_uncertain_detections:
        x = int(det[1])
        y = int(det[2])
        alpha = det[3]
        score = det[4]
        # cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
        image = draw_line(image, (x, y), alpha, line_color=(0, 0, 255), center_color=(0, 0, 0), half_line=True, distance_thresh=40, line_thickness=2)
        cv2.putText(image, str(round(score, 2)), (x + 3, y + 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    for det in image_noisy_detections:
        x = int(det[1])
        y = int(det[2])
        alpha = det[3]
        score = det[4]
        # cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)
        image = draw_line(image, (x, y), alpha, line_color=(255, 0, 0), center_color=(0, 0, 0), half_line=True,
                          distance_thresh=40, line_thickness=2)
        cv2.putText(image, str(round(score, 2)), (x + 3, y + 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
    cv2.imwrite(osp.join(direc, data_loader.image_names[i] + data_loader.ext), image)
