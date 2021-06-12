import numpy as np
import torch
import torchvision
import csv
import os
from os import path as osp
import argparse
import collections
from math import inf
import torch.optim as optim
from torchvision import transforms

import retinanet.utils
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader
from retinanet import csv_eval
from utils.log_utils import log_history
from prediction import imageloader, predict_boxes
import labeling
from retinanet import utils
from retinanet.settings import NAME, X, Y, ALPHA, LABEL
import visualize

parser = argparse.ArgumentParser(description="Get required values for box prediction and labeling.")
parser.add_argument("-f", "--filename-path", required=True, type=str, dest="filenames_path",
                    help="Path to the file that reads the name of image files")
parser.add_argument("-p", "--partition", required=True, type=str, dest="partition",
                    choices=["supervised", "unsupervised", "validation", "test"], help="which part of file names")
parser.add_argument("-c", "--class-list", type=str, required=True, dest="class_list",
                    help="path to the class_list file")

parser.add_argument("-i", "--image-dir", type=str, required=True, dest="image_dir",
                    help="The directory where images are in.")
parser.add_argument("-e", "--extension", type=str, required=False, dest="ext", default=".jpg",
                    choices=[".jpg", ".png"], help="image extension")
parser.add_argument("-m", "--model", required=True, type=str, dest="model",
                    help="path to the model")
parser.add_argument("-a", "--annotations", type=str, required=True, dest="annotations",
                    help="path to the ground_truth annotations compatible with partition")
parser.add_argument("-o", "--output-dir", type=str, required=True, dest="output_dir",
                    help="where to save output")
parser.add_argument("--corrected-path", type=str, required=True, dest="corrected_path",
                    help="path to save corrected annotations")
parser.add_argument("--active-path", type=str, required=True, dest="active_path",
                    help="path to save active annotations")
args = parser.parse_args()

class_to_index, index_to_class = utils.load_classes(csv_class_list_path=args.class_list)


def load_annotations(path: str) -> np.array:
    assert osp.isfile(path), "File does not exist."
    boxes = list()
    fileIO = open(path, "r")
    reader = csv.reader(fileIO, delimiter=",")
    for row in reader:
        if row[X] == row[Y] == row[ALPHA] == row[LABEL] == "":
            continue
        box = [None, None, None, None, None]
        box[NAME] = float(row[NAME])
        box[X] = float(row[X])
        box[Y] = float(row[Y])
        box[ALPHA] = float(row[ALPHA])
        box[LABEL] = float(class_to_index[row[LABEL]])
        boxes.append(box)
    fileIO.close()
    boxes = np.asarray(boxes, dtype=np.float64)
    return np.asarray(boxes[:, [NAME, X, Y, ALPHA, LABEL]], dtype=np.float64)


def detect(dataset, retinanet):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = list()

    retinanet.eval()

    print("detecting")
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            img_name = float(int(data["name"]))

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = retinanet(data['img'].permute(
                    2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = retinanet(
                    data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()
            if boxes.shape[0] == 0:
                continue
            # correct boxes for image scale
            boxes /= scale

            # select detections
            image_boxes = boxes
            image_scores = scores
            image_labels = labels
            img_name_col = np.full(shape=(len(image_scores), 1), fill_value=img_name, dtype=np.int32)
            image_detections = np.concatenate([img_name_col, image_boxes, np.expand_dims(
                image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
            all_detections.extend(image_detections.tolist())
            print('\rimage {0:02d}/{1:02d}'.format(index + 1, len(dataset)), end='')
    print()
    return np.asarray(all_detections, dtype=np.float64)


loader = imageloader.CSVDataset(
    filenames_path=args.filenames_path,
    partition=args.partition,
    class_list=args.class_list,
    images_dir=args.image_dir,
    image_extension=args.ext,
    transform=torchvision.transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
)

model = torch.load(args.model)
pred_boxes = detect(dataset=loader, retinanet=model)
ground_truth_annotations = load_annotations(path=args.annotations)
uncertain_boxes, noisy_boxes = predict_boxes.split_uncertain_and_noisy(pred_boxes)
corrected_boxes = labeling.label(all_gts=ground_truth_annotations, all_uncertain_preds=uncertain_boxes)

noisy_mode = np.full(shape=(noisy_boxes.shape[0], 1), fill_value=retinanet.utils.ActiveLabelMode.noisy.value,
                     dtype=np.float64)
corrected_mode = np.full(shape=(corrected_boxes.shape[0], 1),
                         fill_value=retinanet.utils.ActiveLabelMode.corrected.value, dtype=np.float64)
noisy_boxes = np.concatenate([noisy_boxes[:, [NAME, X, Y, ALPHA, LABEL]], noisy_mode], axis=1)
corrected_boxes = np.concatenate([corrected_boxes[:, [NAME, X, Y, ALPHA, LABEL]], corrected_mode], axis=1)
active_boxes = np.concatenate([corrected_boxes, noisy_boxes], axis=0)
active_boxes = active_boxes[active_boxes[:, NAME].argsort()]

labeling.write_active_boxes(
    boxes=active_boxes,
    path=args.active_path,
    class_dict=index_to_class,
)

labeling.write_corrected_boxes(
    boxes=corrected_boxes[:, [NAME, X, Y, ALPHA, LABEL]],
    path=args.corrected_path,
    class_dict=index_to_class,
)



# boxes = np.concatenate([gt_boxes, uncertain_boxes, noisy_boxes, corrected_boxes], axis=0)
<<<<<<< HEAD
# boxes = np.concatenate([uncertain_boxes, noisy_boxes, corrected_boxes], axis=0)
# boxes = boxes[boxes[:, NAME].argsort()]
# assert osp.isdir(args.output_dir), "Output directory does not exist."
# visualize.draw_noisy_uncertain_gt(loader=loader, detections=boxes, images_dir=args.image_dir, output_dir = args.output_dir)
=======
boxes = np.concatenate([uncertain_boxes, noisy_boxes, corrected_boxes], axis=0)
boxes = boxes[boxes[:, NAME].argsort()]
assert osp.isdir(args.output_dir), "Output directory does not exist."
visualize.draw_noisy_uncertain_gt(loader=loader, detections=boxes, images_dir=args.image_dir, output_dir=args.output_dir)

>>>>>>> dev
