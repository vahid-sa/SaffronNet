import os
import argparse
import torch
from os import path as osp
from torchvision import transforms
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval, my_model as model

class Args:
    def __init__(self):
        self.csv_annotations_path = osp.abspath("./annotations/validation.csv")
        self.state_dict_path = osp.expanduser("~/Saffron/init_fully_trained_weights/init_state_dict.pt")
        self.images_dir = osp.expanduser("~/Saffron/dataset/Train/")
        self.class_list_path = osp.abspath("./annotations/labels.csv")
        self.iou_threshold = 0.5
        self.use_gpu = True


def main(args):
    device = 'cuda' if (torch.cuda.is_available() and args.use_gpu) else 'cpu'
    #dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(
        args.csv_annotations_path,
        args.class_list_path,
        transform=transforms.Compose([Normalizer(), Resizer()]),
        images_dir=args.images_dir,
    )
    # Create the model
    retinanet = model.vgg7(num_classes=dataset_val.num_classes(), pretrained=True)
    # retinanet=torch.load(parser.model_path)

    retinanet = retinanet.to(device=device)
    retinanet = torch.nn.DataParallel(retinanet).to(device=device)
    checkpoint = torch.load(args.state_dict_path)
    retinanet.load_state_dict(checkpoint['model_state_dict'])

    retinanet.training = False
    retinanet.eval()
    # retinanet.module.freeze_bn()

    print("Average precision:", csv_eval.evaluate(dataset_val, retinanet))


if __name__ == '__main__':
    arguments = Args()
    main(args=arguments)
