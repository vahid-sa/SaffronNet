import sys
import os
import torch
import shutil
from os import path as osp
from torchvision import transforms
from torch.utils.data import DataLoader
sys.path.append(osp.abspath("../"))
sys.path.append(osp.abspath("./"))
import retinanet
from retinanet import my_model as model
from retinanet import dataloader
from prediction import imageloader
from utils.active_tools import UncertaintyStatus, ActiveStatus, write_active_annotations, write_corrected_annotations


from cv2 import cv2
import numpy as np

retinanet.settings.NUM_QUERIES = 100
retinanet.settings.NOISY_THRESH = 0.15


class parser:
    ground_truth_annotations = osp.abspath("annotations/supervised.csv")
    csv_classes = osp.abspath("annotations/labels.csv")
    images_dir = osp.expanduser("~/Saffron/dataset/Train/")
    corrected_annotations = osp.expanduser("~/Saffron/active_annotations/corrected.csv")
    filenames = osp.abspath("annotations/filenames.json")
    partition = "supervised"
    ext = ".jpg"
    model_path = osp.expanduser('~/Saffron/weights/supervised/init_model.pt')
    state_dict_path = osp.expanduser('~/Saffron/weights/supervised/init_state_dict.pt')
    states_dir = osp.expanduser("~/Saffron/active_annotations/states")
    active_annotations = osp.expanduser("~/Saffron/active_annotations/train.csv")
    save_directory = osp.expanduser("~/tmp/saffron_imgs/")

    @staticmethod
    def reset():
        if osp.isfile(parser.corrected_annotations):
            os.remove(parser.corrected_annotations)
        if osp.isfile(parser.active_annotations):
            os.remove(parser.active_annotations)
        if osp.isdir(parser.states_dir):
            shutil.rmtree(parser.states_dir)
        os.makedirs(parser.states_dir, exist_ok=False)
        if osp.isdir(parser.save_directory):
            shutil.rmtree(parser.save_directory)
        os.makedirs(parser.save_directory, exist_ok=False)


parser.reset()
retinanet_model = torch.load(parser.model_path)
image_loader = imageloader.CSVDataset(
    filenames_path=parser.filenames,
    partition=parser.partition,
    class_list=parser.csv_classes,
    images_dir=parser.images_dir,
    image_extension=parser.ext,
    transform=transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
)


uncertainty_status = UncertaintyStatus(
    loader=image_loader,
    model=retinanet_model,
    class_list_file_path=parser.csv_classes,
    corrected_annotations_file_path=parser.corrected_annotations,
)


data_loader = dataloader.CSVDataset(
    train_file=parser.ground_truth_annotations,
    class_list=parser.csv_classes,
    images_dir=parser.images_dir,
    transform=transforms.Compose([dataloader.Normalizer(), dataloader.Resizer()]),
)


images_detections = uncertainty_status.get_active_predictions()
uncertainty_status.load_uncertainty_states(boxes=images_detections)
uncertainty_status.write_states(directory=parser.states_dir)
uncertain_detections = images_detections["uncertain"]
noisy_detections = images_detections["noisy"]

active_status = ActiveStatus(data_loader=data_loader)
corrected_annotations = active_status.correct(uncertainty_states=uncertainty_status.tile_states)
active_annotations = ActiveStatus.concat_noisy_and_corrected_boxes(corrected_boxes=corrected_annotations, noisy_boxes=noisy_detections)
active_states = uncertainty_status.tile_states

write_corrected_annotations(annotations=corrected_annotations, path=parser.corrected_annotations, class_list_path=parser.csv_classes)
write_active_annotations(annotations=active_annotations, path=parser.active_annotations, class_list_path=parser.csv_classes)


dataset_train = dataloader.CSVDataset(
    train_file=parser.active_annotations,
    class_list=parser.csv_classes,
    images_dir=parser.images_dir,
    ground_truth_states_directory=parser.states_dir,
    transform=transforms.Compose([dataloader.Normalizer(), dataloader.Augmenter(), dataloader.Resizer()]),
    save_output_img_directory=parser.save_directory
)

sampler = dataloader.AspectRatioBasedSampler(
    dataset_train, batch_size=1, drop_last=False)
dataloader_train = DataLoader(
    dataset_train, num_workers=2, collate_fn=dataloader.collater, batch_sampler=sampler)

retinanet = model.vgg7(num_classes=dataset_train.num_classes(), pretrained=True)
if torch.cuda.is_available():
    retinanet = torch.nn.DataParallel(retinanet.cuda()).cuda()
else:
    retinanet = torch.nn.DataParallel(retinanet)

retinanet.training = True
optimizer = torch.optim.Adam(retinanet.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

checkpoint = torch.load(parser.state_dict_path)
retinanet.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

print('Num training images: {}'.format(len(dataset_train)))
retinanet.train()
optimizer.zero_grad()
unnormalizer = dataloader.UnNormalizer()
for i, data in enumerate(dataloader_train):
    model_directory = osp.join(parser.save_directory, "model")
    os.makedirs(model_directory, exist_ok=True)
    params = [data['img'].cuda().float(), data['annot'], data['gt_state'].cuda(), data["aug_img_path"], model_directory]
    classification_loss, xydistance_regression_loss, angle_distance_regression_losses = retinanet(params)
