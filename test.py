import os
import numpy as np
import cv2
import torchvision
import csv
from os import path as osp
from retinanet.dataloader import CSVDataset, Normalizer, Resizer
from retinanet.utils import load_classes


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
        self.canvas = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)

    def __call__(self, index, color):
        left, top, right, bottom = index
        self.canvas[top:bottom, left:right] = color


def tile(image, size):
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





"""
img = cv2.imread("/home/spx/Documents/saffron_dataset/Train/001.jpg")
img = pad_image(img)
indices = tile(image=img, size=100)
colors = np.random.randint(low=0, high=256, size=len(indices), dtype=np.uint8)
mask = Mask(image=img)
tuple(mask(index, color) for index, color in zip(indices, colors))
cv2.imshow("canvas", mask.canvas)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
images_dir = osp.expanduser("~/Documents/saffron_dataset/Train")
csv_classes = osp.abspath("annotations/labels.csv")
csv_anots = osp.abspath("annotations/test.csv")
dataset = Dataset(csv_annotations_path=csv_anots, labels_path=csv_classes, images_dir=images_dir)
dataset.column_wise = False

data = dataset[5]
img, annots = data['img'], data['annots']
canvas = np.zeros(shape=img.shape[:2], dtype=np.float64)
indices = tile(image=img, size=100)
status = np.zeros(shape=len(indices), dtype=np.uint8)
for annot in annots:
    if annot['is_asked']:
        state = list(np.logical_and(np.logical_and(np.logical_and(indices[:, 0] <= annot['x'], indices[:, 1] <= annot['y']), indices[:, 2] >= annot['x']), indices[:, 3] >= annot['y']))
        status[state.index(True)] = 255
mask = Mask(image=img)
tuple(mask(index, state) for index, state in zip(indices, status))
cv2.imshow("canvas", mask.canvas)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# dataset = CSVDataset(
#     train_file=csv_anots,
#     class_list=csv_classes,
#     images_dir=images_dir,
#     transform=torchvision.transforms.Compose([Normalizer(), Resizer()]),
# )


