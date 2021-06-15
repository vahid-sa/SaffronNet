"""
Cluster flowers based on assign anchors according to angle.
"""
import argparse

from retinanet.dataloader import CSVDataset
from torch.utils import data
import matplotlib.pyplot as plt


def setup_parser(args):
    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')

    parser.add_argument(
        '--csv_classes', help='Path to file containing class list (see readme)')

    parser.add_argument(
        '--csv_anots', help='Path to file containing list of anotations')

    parser.add_argument(
        '--images_dir', help='images base folder')

    parser.add_argument(
        '--save_dir', help='output directory for generated images')

    parser = parser.parse_args(args)

    return parser


def main(args=None):
    parser = setup_parser(args)
    dataset = CSVDataset(
        images_dir=parser.images_dir,
        train_file=parser.csv_anots,
        class_list=parser.csv_classes
    )
    for image_index in range(len(dataset)):
        image = dataset.load_image(image_index)
        annots = dataset.load_annotations(image_index)


if __name__ == "__main__":
    main()
