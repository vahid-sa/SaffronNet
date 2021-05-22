import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
import json
import torchvision
from retinanet.settings import NUM_VARIABLES
from utils.visutils import draw_line
from retinanet import imageloader


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError(
                'line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError(
                'line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def detect_image(image_dir, filenames, model_path, class_list, output_dir, ext=".jpg"):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    for img_name in filenames:

        image = cv2.imread(os.path.join(image_dir, img_name+ext))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros(
            (rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))
        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()

            scores, classification, transformed_anchors = model(
                image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            transformed_anchors = transformed_anchors.cpu().detach().numpy()
            for j in range(idxs[0].shape[0]):
                center_alpha = transformed_anchors[idxs[0][j], :]
                x, y, alpha = int(center_alpha[0]), int(
                    center_alpha[1]), int(center_alpha[2])
                # label_name = labels[int(classification[idxs[0][j]])]
                score = scores[j]
                # caption = '{} {:.3f}'.format(label_name, score)
                image_orig = draw_line(
                    image=image_orig,
                    p=(x, y),
                    alpha=alpha,
                    line_color=(0, 255, 0),
                    center_color=(0, 0, 255),
                    half_line=True)
            cv2.imwrite(
                os.path.join(output_dir, "{0}.jpg".format(img_name)), image_orig)


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [None for j in range(len(dataset))]

    retinanet.eval()

    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

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

            # correct boxes for image scale
            boxes /= scale

            # select detections
            image_boxes = boxes
            image_scores = scores
            image_labels = labels
            img_name = int(data["name"])
            img_name_col = np.full(shape=(len(image_scores), 1), fill_value=img_name, dtype=np.int32)
            image_detections = np.concatenate([img_name_col, image_boxes, np.expand_dims(
                image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
            all_detections[index] = image_detections

            print('{}/{}'.format(index + 1, len(dataset)))

    return all_detections


def draw(loader, detections, images_dir, output_dir, ext=".jpg"):
    assert len(loader) == len(detections), "detections and loader images must be same length"
    unnormalize = imageloader.UnNormalizer()
    for i in range(len(loader)):
        img_path = os.path.join(images_dir, "{0}{1}".format(loader[i]["name"], ext))
        img = cv2.imread(img_path)
        detection = detections[i][0]
        print("detection:\n",detection)
        for j in range(len(detection)):
            det = detection[j]
            im_name, x, y, alpha, score, label = det
            img = draw_line(
                image=img,
                p=(x, y),
                alpha=90.0 - alpha,
                line_color=(0, 255, 0),
                center_color=(0, 0, 255),
                half_line=True)
        save_path = os.path.join(output_dir, loader[i]["name"] + ext)
        cv2.imwrite(save_path, img)
        print("saved", i)


def detect_draw(filenames_path, partition, class_list, images_dir, output_dir, model_path, image_extension=".jpg"):
    transform = torchvision.transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()])
    loader = imageloader.CSVDataset(
        filenames_path=filenames_path,
        partition=partition,
        class_list=class_list,
        images_dir=images_dir,
        image_extension=image_extension,
        transform=transform,
    )
    retinanet = torch.load(model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    all_detections = _get_detections(loader, retinanet)
    draw(loader=loader, detections=all_detections, output_dir=output_dir, images_dir=images_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Simple script for visualizing result of training.')

    parser.add_argument(
        '--image_dir', help='Path to directory containing images')
    parser.add_argument(
        '--model_path', help='Path to model')
    parser.add_argument(
        "--path_mod", help="supervised | unsupervised | validation | test")
    parser.add_argument(
        '--class_list', help='Path to CSV file listing class names (see README)')
    parser.add_argument(
        '--output_dir', help='direction for output images')

    parser = parser.parse_args()

    with open("annotations/filenames.json", "r") as fileIO:
        str_names = fileIO.read()
    names = json.loads(str_names)
    assert parser.path_mod in "supervised | unsupervised | validation | test"

    detect_draw(
        filenames_path="annotations/filenames.json",
        partition=parser.path_mod,
        class_list=parser.class_list,
        output_dir=parser.output_dir,
        images_dir=parser.image_dir,
        model_path=parser.model_path,
    )
    # detect_image(
    #     image_dir=parser.image_dir,
    #     filenames=names[parser.path_mod],
    #     model_path=parser.model_path,
    #     class_list=parser.class_list,
    #     output_dir=parser.output_dir
    # )
