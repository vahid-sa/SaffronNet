import sys
import os
import torch
import torchvision
import unittest
import argparse
from os import path as osp

PACKAGE_PARENT = '..'
SCRIPT_DIR = osp.dirname(os.path.realpath(osp.join(os.getcwd(), osp.expanduser(__file__))))
sys.path.append(osp.normpath(osp.join(SCRIPT_DIR, PACKAGE_PARENT)))

from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval


class TestModel(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.args = args

        self.dataset_val = CSVDataset(
            train_file="./annotations/validation.csv",
            class_list="./annotations/labels.csv",
            transform=torchvision.transforms.Compose([Normalizer(), Resizer()]),
            images_dir=self.args.image_dir,
            image_extension=self.args.ext,
        )

    def test_model(self):
        fileIO = open(self.args.model, "rb")
        loaded_model = torch.load(fileIO)
        fileIO.close()
        model = loaded_model.cuda() if torch.cuda.is_available() else loaded_model
        model.training = False
        model.eval()
        mAP = csv_eval.evaluate(self.dataset_val, model)
        print(mAP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test load model")
    parser.add_argument("-m", "--model", required=True, type=str, dest="model",
                        help="path to the model")
    parser.add_argument("-e", "--extension", type=str, required=False, dest="ext", default=".jpg",
                        choices=[".jpg", ".png"], help="image extension")
    parser.add_argument("-i", "--image-dir", type=str, required=True, dest="image_dir",
                        help="The directory where images are in.")
    args = parser.parse_args()
    unittest.main()
