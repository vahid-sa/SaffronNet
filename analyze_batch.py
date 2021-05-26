import retinanet
from typing import List
import numpy as np
from typing import List
import argparse
import os
import json
import cv2 as cv
from numpy.core.fromnumeric import std
from retinanet.csv_eval import evaluate
from utils.visutils import DrawMode, std_draw_line
from retinanet.settings import NUM_VARIABLES
from retinanet.dataloader import CSVDataset, Normalizer, Resizer
from torchvision import transforms
import torch


def analysis_visualizer(image: np.ndarray, image_name: str, accepted_predictions: List,
                        declined_predictions: List, annotations: List, write_dir: str):

    annotations = annotations[:, :NUM_VARIABLES]
    for anot in annotations:
        x, y, alpha = anot
        image = std_draw_line(
            image=image, point=(x, y), alpha=90-alpha, mode=DrawMode.Raw)

    accepted_predictions = accepted_predictions[:, :NUM_VARIABLES]
    for pred in accepted_predictions:
        x, y, alpha = pred
        image = std_draw_line(
            image=image, point=(x, y), alpha=90-alpha, mode=DrawMode.Accept)

    declined_predictions = declined_predictions[:, :NUM_VARIABLES]
    for pred in declined_predictions:
        x, y, alpha = pred
        image = std_draw_line(
            image=image, point=(x, y), alpha=90-alpha, mode=DrawMode.Decline)

    cv.imwrite(
        os.path.join(write_dir, '{}.jpg'.format(image_name), image))
    data = {}
    data['num accepted anchors'] = len(accepted_predictions)
    data['num declined anchors'] = len(declined_predictions)
    data['accepted anchors'] = accepted_predictions
    data['declined anchors'] = declined_predictions

    with open(os.path.join(write_dir, '{}.json'.format(image_name)), 'w') as outfile:
        json.dump(data, outfile)


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Simple analizing script for training a RetinaNet network.')

    parser.add_argument('--images_dir', help='image files direction', type=str)
    parser.add_argument(
        '--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument(
        '--csv_test', help='Path to file containing training annotations (see readme)')
    parser.add_argument(
        '--visualize_path', help="save visualized images in visualize_path")
    parser.add_argument(
        '--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument(
        '--model_path', help='model path')
    parser.add_argument(
        '--ext', help='image file extention', type=str, default='.jpg')

    parser.add_argument(
        '--write_dir', help='write dir for analysys', type=str)

    parser = parser.parse_args(args)

    generator = CSVDataset(train_file=parser.csv_test, class_list=parser.csv_classes,
                           transform=transforms.Compose([Normalizer(), Resizer()]), images_dir=parser.images_dir, image_extension=parser.ext)

    model = torch.load(parser.model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    evaluate(
        generator=generator,
        retinanet=model,
        visualizer=analysis_visualizer,
        write_dir=parser.write_dir
    )


if __name__ == "__main__":
    main()
