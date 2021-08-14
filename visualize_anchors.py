from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet.losses import calc_distance
from retinanet.anchors import Anchors
from utils.visutils import draw_line
from torchvision import transforms
import numpy as np
import os
from os import path as osp
import argparse
import torch
from retinanet.settings import *
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def visualize_anchors(anchors, annots, load_image_path, save_image_dir, targets):
    image = cv2.cvtColor(cv2.imread(load_image_path), cv2.COLOR_BGR2RGB)
    _anchors = anchors[0, :, :]
    for anchor in _anchors[targets.squeeze() == 1]:
        x, y, alpha = anchor[0], anchor[1], 90 - anchor[2]
        image = draw_line(
            image, (x, y), alpha,
            line_color=(0, 255, 0),
            center_color=(0, 0, 255),
            half_line=True,
            distance_thresh=40,
            line_thickness=2
        )
    for anot in annots:
        x, y, alpha = anot[0], anot[1], 90 - anot[2]
        image = draw_line(
            image, (x, y), alpha,
            line_color=(0, 0, 0),
            center_color=(255, 0, 0),
            half_line=True
        )
    for anchor in _anchors[targets.squeeze() == -1]:
        x, y, alpha = anchor[0], anchor[1], 90 - anchor[2]
        image = draw_line(
            image, (x, y), alpha,
            line_color=(255, 255, 0),
            center_color=(0, 0, 255),
            half_line=True,
            distance_thresh=40,
            line_thickness=2

        )
    save_image_path = osp.join(save_image_dir, osp.basename(load_image_path))
    cv2.imwrite(save_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')

    parser.add_argument(
        '--dataset', help='Dataset type, must be one of csv or coco.')

    parser.add_argument(
        '--csv_classes', help='Path to file containing class list (see readme)')

    parser.add_argument(
        '--csv_anots', help='Path to file containing list of anotations'
    )

    parser.add_argument(
        '--images_dir', help='images base folder'
    )

    parser.add_argument(
        '--save_dir', help='output directory for generated images'
    )

    parser = parser.parse_args(args)

    if parser.dataset == 'csv':
        dataset = CSVDataset(train_file=parser.csv_anots, class_list=parser.csv_classes, images_dir=parser.images_dir,
                             transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError(
            'Dataset type not understood (must be csv or coco), exiting.')

    sample_image = (dataset.load_image(0) * 255).astype(np.int32)
    sample_batch = np.expand_dims(sample_image, axis=0)
    sample_batch = sample_batch.transpose(0, 3, 1, 2)
    anchros_mudole = Anchors()
    anchors = anchros_mudole(sample_batch)

    for i in range(len(dataset)):
        image = (dataset.load_image(i) * 255).astype(np.int32)
        anots = dataset.load_annotations(i)

        anchors = torch.tensor(anchors)
        anots = torch.tensor(anots)
        if torch.cuda.is_available():
            anchors = anchors.cuda()
            anots = anots.cuda()
        dxy, dalpha = calc_distance(anchors[0, :, :], anots[:, :NUM_VARIABLES])

        dxy_min, dxy_argmin = torch.min(dxy, dim=1)  # num_anchors x 1

        # compute the loss for classification
        targets = torch.ones(anchors.shape[1], 1) * -1
        if torch.cuda.is_available():
            targets = targets.cuda()
# -----------------------------------------------------------------------

        targets[torch.ge(
            dxy_min, 1.5 * MAX_ANOT_ANCHOR_POSITION_DISTANCE), :] = 0

        print('dalpha.shape: ', dalpha.shape)
        print('dxy_argmin.shape: ', dxy_argmin.shape)
        print('dxy_min.shape: ', dxy_min.shape)
        a = dalpha[range(dalpha.shape[0]), dxy_argmin]
        print('a.shape: ', a.shape)
        print('a: ', a[:20])
        targets[torch.ge(
            a, 1.5 * MAX_ANOT_ANCHOR_ANGLE_DISTANCE), :] = 0

        positive_indices = torch.logical_and(
            torch.le(
                dxy_min, MAX_ANOT_ANCHOR_POSITION_DISTANCE),
            torch.le(
                a, MAX_ANOT_ANCHOR_ANGLE_DISTANCE
            )
        )
        print('positive_indices.shape: ', positive_indices.shape)

        d_argmin = positive_indices.nonzero(as_tuple=True)[0]
        d_argmin = dxy_argmin[d_argmin]

        print('d_argmin.shape: ', d_argmin.shape)
        print('d_argmin: ', d_argmin[:50])
        num_positive_anchors = positive_indices.sum()

        # assigned_annotations = center_alpha_annotation[deltaphi_argmin, :] # no different in result
        assigned_annotations = anots[d_argmin, :]
        print('_anots: ', anots[:10])
        targets[positive_indices, :] = 0

        targets[positive_indices,
                assigned_annotations[d_argmin, 3].long()] = 1

        _anchors = anchors[0, :, :]
        for anchor in _anchors[targets.squeeze() == 1]:
            x, y, alpha = anchor[0], anchor[1], 90 - anchor[2]
            image = draw_line(
                image, (x, y), alpha,
                line_color=(0, 255, 0),
                center_color=(0, 0, 255),
                half_line=True,
                distance_thresh=40,
                line_thickness=2
            )
        for anot in anots:
            x, y, alpha = anot[0], anot[1], 90 - anot[2]
            image = draw_line(
                image, (x, y), alpha,
                line_color=(0, 0, 0),
                center_color=(255, 0, 0),
                half_line=True
            )
        for anchor in _anchors[targets.squeeze() == -1]:
            x, y, alpha = anchor[0], anchor[1], 90 - anchor[2]
            image = draw_line(
                image, (x, y), alpha,
                line_color=(255, 255, 0),
                center_color=(0, 0, 255),
                half_line=True,
                distance_thresh=40,
                line_thickness=2

            )
        image_name = os.path.basename(dataset.image_names[i])
        cv2.imwrite(os.path.join(parser.save_dir, image_name),
                    cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
