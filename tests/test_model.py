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
from retinanet import csv_eval, model


class TestModel:
    def __init__(self):
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
        loaded_model = loaded_model.cuda() if torch.cuda.is_available() else loaded_model
        loaded_model.training = False
        loaded_model.eval()
        mAP = csv_eval.evaluate(self.dataset_val, loaded_model)
        print("model", mAP)

    def test_state_dict(self):
        fileIO = open(self.args.state_dict, "rb")
        checkpoint = torch.load(fileIO)
        fileIO.close()

        defined_model = model.resnet50(
            num_classes=self.dataset_val.num_classes(), pretrained=True)

        if torch.cuda.is_available():
            retinanet = torch.nn.DataParallel(defined_model.cuda()).cuda()
        else:
            retinanet = torch.nn.DataParallel(defined_model)

        optimizer = torch.optim.Adam(retinanet.parameters(), lr=1e-5)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        retinanet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        retinanet.training = False
        retinanet.eval()
        retinanet.module.freeze_bn()
        mAP = csv_eval.evaluate(self.dataset_val, retinanet)
        print("state_dict", mAP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test load model")
    parser.add_argument("-m", "--model", required=True, type=str, dest="model",
                        help="path to the model")
    parser.add_argument("-e", "--extension", type=str, required=False, dest="ext", default=".jpg",
                        choices=[".jpg", ".png"], help="image extension")
    parser.add_argument("-s", "--state-dict", required=True, type=str, dest="state_dict",
                        help="path to the state_dict")
    parser.add_argument("-i", "--image-dir", type=str, required=True, dest="image_dir",
                        help="The directory where images are in.")
    args = parser.parse_args()
    tm = TestModel()
    tm.test_model()
    tm.test_state_dict()
