import os
import shutil
import logging
from numpy.random.mtrand import sample
import torch
import numpy as np
import csv
import glob
import json
import cv2
from os import path as osp
from torch._C import TracingState
from torchvision import transforms

from retinanet import model, dataloader
from retinanet.utils import ActiveLabelMode, load_classes, ActiveLabelModeSTR
from utils.prediction import detect
from prediction import imageloader
from utils.active_tools import Active
from retinanet.utils import unnormalizer
from utils.visutils import draw_line

device = "cuda" if torch.cuda.is_available() else "cpu"

loader = imageloader.CSVDataset(
    filenames_path=osp.abspath("annotations/filenames.json"),
    partition="unsupervised",
    class_list=osp.abspath("annotations/labels.csv"),
    images_dir=osp.expanduser("~/Saffron/dataset/Train/"),
    transform=transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
)

gt_loader = dataloader.CSVDataset(
    train_file=osp.abspath("./annotations/unsupervised.csv"),
    class_list=osp.abspath("./annotations/labels.csv"),
    images_dir=osp.expanduser("~/Saffron/dataset/Train/"),
    transform=transforms.Compose([dataloader.Normalizer(), dataloader.Resizer()]),
)

retinanet = model.vgg7(num_classes=1, pretrained=True)
retinanet = retinanet.to(device=device)
retinanet = torch.nn.DataParallel(retinanet)
retinanet = retinanet.to(device=device)
checkpoint = torch.load(osp.expanduser('~/Saffron/init_fully_trained_weights/init_state_dict.pt'))
retinanet.load_state_dict(checkpoint['model_state_dict'])
retinanet.eval()
retinanet.training = False
annot_path = osp.expanduser("~/Saffron/active_annotations.csv")
if osp.isfile(annot_path):
    os.remove(annot_path)
shutil.copyfile(osp.abspath("./annotations/supervised.csv"), annot_path)
active = Active(
    loader=loader,
    annotations_path=annot_path,
    class_list_path=osp.abspath("./annotations/labels.csv"),
    budget=100,
    ground_truth_dataloader=gt_loader,
    aggregator_type="sum",
    uncertainty_algorithm="least",
)

active.create_annotations(model=retinanet)

pred_boxes = active.predictions
uncertain_imgs = active.uncertain_images
gt_img_names = [int(osp.splitext(osp.basename(name))[0]) for name in gt_loader.image_names]

write_dir = osp.expanduser("~/st/Saffron/tmp/")
if osp.isdir(write_dir):
    shutil.rmtree(write_dir)
os.makedirs(write_dir)
active_loader = dataloader.CSVDataset(
    train_file=annot_path,
    class_list=osp.abspath("./annotations/labels.csv"),
    images_dir=osp.expanduser("~/Saffron/dataset/Train/"),
    transform=transforms.Compose([dataloader.Normalizer(), dataloader.Resizer()]),
)
for index, sample in enumerate(active_loader):
    img = sample['img']
    img = unnormalizer(img.detach().cpu().numpy())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_boxes = pred_boxes[pred_boxes[:, 0] == int(sample['name'])]
    # annots = sample['annot'].detach().cpu().numpy()
    for box in img_boxes:
        x, y, alpha = int(box[1]), int(box[2]), int(box[3])
        score = box[5]
        img = draw_line(img, (x, y), alpha, line_color=(0, 0, 255), center_color=(0, 0, 0), half_line=True,
                        distance_thresh=40, line_thickness=2)
    for annot in sample["annot"]:
        x, y, alpha = annot[0], annot[1], annot[2]
        img = draw_line(img, (x, y), alpha, line_color=(0, 0, 0), center_color=(0, 0, 0), half_line=True,
                        distance_thresh=40, line_thickness=2)
        # cv2.putText(img, str(round(score, 2)), (x + 3, y + 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    write_path = osp.join(write_dir, f"{sample['name']}.jpg")
    cv2.imwrite(write_path, img)

f = open(annot_path, "r")
reader = csv.reader(f)
count = 0
for _ in reader:
    count += 1
f.close()
