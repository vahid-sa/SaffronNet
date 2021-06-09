import argparse
import collections
from math import inf

import numpy as np
import os
import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval
from utils.log_utils import log_history

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    max_mAp = 0
    min_loss = inf
    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')

    parser.add_argument(
        '--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument(
        '--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument(
        '--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument(
        '--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument(
        '--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

    parser.add_argument(
        '--ext', help='image file extention', type=str, default='.jpg')

    parser.add_argument('--images_dir', help='image files direction', type=str)
    parser.add_argument('--save_dir', help='model save dir', type=str)
    parser.add_argument('--epochs', help='Number of epochs',
                        type=int, default=100)

    parser.add_argument('--resume', help='flag for resume training',
                        type=bool, default=False)
    parser.add_argument(
        '--model_path', help='path for saved state dict to resuming model')

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError(
                'Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]), images_dir=parser.images_dir, image_extension=parser.ext)

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]), images_dir=parser.images_dir, image_extension=parser.ext)

    else:
        raise ValueError(
            'Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(
        dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(
        dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(
            dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(
            dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(
            num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError(
            'Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True)

    loss_hist = []
    mAp_hist = []

    if parser.resume:
        checkpoint = torch.load(parser.model_path)
        retinanet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    classification_loss, xydistance_regression_loss, angle_distance_regression_losses = retinanet(
                        [data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, xydistance_regression_loss, angle_distance_regression_losses = retinanet(
                        [data['img'].float(), data['annot']])
                classification_loss = classification_loss.mean()
                xydistance_regression_loss = xydistance_regression_loss.mean()
                angle_distance_regression_losses = angle_distance_regression_losses.mean()

                loss = classification_loss + xydistance_regression_loss + \
                    angle_distance_regression_losses

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | XY Regression loss: {:1.5f} | Angle Regression loss: {:1.5f}| Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(xydistance_regression_loss), float(angle_distance_regression_losses), np.mean(loss)))

                del classification_loss
                del xydistance_regression_loss
                del angle_distance_regression_losses

            except Exception as e:
                print(e)
                continue

        mAP = None
        if parser.dataset == 'coco':
            print('Evaluating dataset')
            coco_eval.evaluate_coco(dataset_val, retinanet)
        elif parser.dataset == 'csv' and parser.csv_val is not None:
            mean_epoch_loss = np.mean(epoch_loss)
            print('Evaluating dataset')
            if min_loss > mean_epoch_loss:
                print("loss improved from from {} to {}".format(
                    min_loss, mean_epoch_loss))
                min_loss = mean_epoch_loss
                if parser.save_dir:
                    PATH = os.path.join(parser.save_dir, 'best_model_loss.pt')
                else:
                    PATH = 'best_model_loss.pt'
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': retinanet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': 'Running loss: {:1.5f}'.format(np.mean(epoch_loss))
                }, PATH)

            mAP = csv_eval.evaluate(dataset_val, retinanet)
            if mAP[0][0] > max_mAp:
                print('mAp improved from {} to {}'.format(max_mAp, mAP[0][0]))
                max_mAp = mAP[0][0]
                if parser.save_dir:
                    PATH = os.path.join(parser.save_dir, 'best_model_mAp.pt')
                else:
                    PATH = 'best_model_mAp.pt'
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': retinanet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': np.mean(epoch_loss),
                    'mAp': max_mAp
                }, PATH)
                torch.save(retinanet, os.path.join(os.path.dirname(
                    PATH), 'best_model_mAp_ready_to_eval.pt'))

        log_history(epoch_num, {'loss': np.mean(epoch_loss), 'mAp': mAP}, os.path.join(
            os.path.dirname(PATH), 'history.json'))
        scheduler.step(np.mean(epoch_loss))

    retinanet.eval()
    if parser.save_dir:
        torch.save(retinanet, os.path.join(parser.save_dir, 'model_final.pt'))
    else:
        torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
