import os
import cv2
import shutil
import torch
import torchvision
import numpy as np
from os import path as osp


from retinanet import model
from retinanet import dataloader
from retinanet import utils
from prediction import imageloader
from utils.active_tools import Active
from utils.visutils import draw_line, Visualizer

class Args:
    ground_truth_annotations = osp.abspath("annotations/unsupervised.csv")
    csv_classes = osp.abspath("annotations/labels.csv")
    images_dir = osp.expanduser("~/Saffron/dataset/Train/")
    corrected_annotations = osp.expanduser("~/Saffron/active_annotations/corrected.csv")
    filenames = osp.abspath("annotations/filenames.json")
    partition = "unsupervised"
    ext = ".jpg"
    state_dict_path = osp.expanduser('~/Saffron/init_fully_trained_weights/init_state_dict.pt')
    states_dir = osp.expanduser("~/Saffron/active_annotations/states")
    active_annotations = osp.expanduser("~/Saffron/active_annotations/train.csv")
    save_directory = osp.expanduser("~/st/Saffron/imgs/")
    epochs = 30
    csv_val = osp.abspath("./annotations/validation.csv")
    save_models_directory = osp.expanduser("~/st/Saffron/weights/active")
    cycles = 10
    budget = 100
    supervised_annotations = osp.abspath("./annotations/supervised.csv")
    metrics_path = osp.expanduser("~/st/Saffron/metrics.json")
    uncertainty_algorithm = "least"

    @staticmethod
    def reset():
        if osp.isfile(Args.corrected_annotations):
            os.remove(Args.corrected_annotations)
        if osp.isfile(Args.active_annotations):
            os.remove(Args.active_annotations)
        if osp.isdir(Args.states_dir):
            shutil.rmtree(Args.states_dir)
        os.makedirs(Args.states_dir, exist_ok=False)
        if osp.isdir(Args.save_directory):
            shutil.rmtree(Args.save_directory)
        os.makedirs(Args.save_directory, exist_ok=False)
        if osp.isdir(Args.save_models_directory):
            shutil.rmtree(Args.save_models_directory)
        os.makedirs(Args.save_models_directory, exist_ok=False)
        if osp.isfile(Args.metrics_path):
            os.remove(Args.metrics_path)

Args.reset()

image_loader = imageloader.CSVDataset(
    filenames_path=Args.filenames,
    partition="unsupervised",
    class_list=Args.csv_classes,
    images_dir=Args.images_dir,
    image_extension=".jpg",
    transform=torchvision.transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
)

gt_loader = dataloader.CSVDataset(
    train_file=Args.ground_truth_annotations,
    class_list=Args.csv_classes,
    images_dir=Args.images_dir,
    transform=torchvision.transforms.Compose([dataloader.Normalizer(), dataloader.Resizer()]),
)

active = Active(
    loader=image_loader,
    states_dir=Args.states_dir,
    radius=50,
    image_string_file_numbers_path=Args.filenames,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
retinanet = model.vgg7(num_classes=1, pretrained=True)
retinanet = retinanet.to(device=device)
retinanet = torch.nn.DataParallel(retinanet)
retinanet = retinanet.to(device=device)

optimizer = torch.optim.Adam(retinanet.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

checkpoint = torch.load(Args.state_dict_path)
retinanet.load_state_dict(checkpoint['model_state_dict'])

active.create_active_annotations(
    model=retinanet,
    budget=Args.budget,
    ground_truth_loader=gt_loader,
    ground_truth_annotations_path=Args.corrected_annotations,
    active_annotations_path=Args.active_annotations,
    classes_list_path=Args.csv_classes,
)

uncertain_detections = active.uncertain_predictions
active_annotations = active.active_annotations
corrected_annotations = active.ground_truth_annotations
noisy_detections = active.noisy_predictions
noisy_detections = active.noisy_predictions
for i in range(len(image_loader.image_names)):
    image = cv2.imread(osp.join(Args.images_dir, image_loader.image_names[i] + Args.ext))
    mask = np.full(fill_value=0.5, dtype=np.float64, shape=image.shape)
    mask[active.states[i]] = 1.0
    image = np.multiply(image.astype(np.float64), mask).astype(np.float32)
    image_noisy_detections = noisy_detections[noisy_detections[:, 0] == int(image_loader.image_names[i])]
    image_uncertain_detections = uncertain_detections[uncertain_detections[:, 0] == int(image_loader.image_names[i])]
    image_active_annotations = active_annotations[active_annotations[:, 0] == int(image_loader.image_names[i])]
    image_corrected_annotations = corrected_annotations[corrected_annotations[:, 0] == int(image_loader.image_names[i])]
    gt_annotations = gt_loader[i]['annot'].detach().cpu().numpy()
    for annotation in gt_annotations:
        x = annotation[0]
        y = annotation[1]
        alpha = annotation[2]
        image = draw_line(image, (x, y), alpha, line_color=(0, 0, 0), center_color=(0, 0, 0), half_line=True,
                        distance_thresh=40, line_thickness=2)

    for det in image_uncertain_detections:
        x = int(det[1])
        y = int(det[2])
        alpha = det[3]
        score = det[5]
        image = draw_line(image, (x, y), alpha, line_color=(0, 0, 255), center_color=(0, 0, 0), half_line=True, distance_thresh=40, line_thickness=2)
        cv2.putText(image, str(round(score, 2)), (x + 3, y + 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    for det in image_noisy_detections:
        x = int(det[1])
        y = int(det[2])
        alpha = det[3]
        score = det[5]
        image = draw_line(image, (x, y), alpha, line_color=(255, 0, 0), center_color=(0, 0, 0), half_line=True,
                        distance_thresh=40, line_thickness=2)
        cv2.putText(image, str(round(score, 2)), (x + 3, y + 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

    for annot in image_corrected_annotations:
        x = int(annot[1])
        y = int(annot[2])
        alpha = annot[3]
        score = annot[5]
        image = draw_line(image, (x, y), alpha, line_color=(0, 255, 0), center_color=(0, 0, 0), half_line=True,
                        distance_thresh=40, line_thickness=2)

    write_path = osp.join(Args.save_directory, image_loader.image_names[i] + Args.ext)
    cv2.imwrite(write_path, image)
