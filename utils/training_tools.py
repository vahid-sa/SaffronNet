import logging
import os
import shutil
from typing import List
import torch
import numpy as np
import csv
import cv2
import json
from os import path as osp
from math import inf
from torchvision import transforms
from torch.utils.data import DataLoader
from retinanet import dataloader, csv_eval
from retinanet import model
from prediction import imageloader
from utils.active_tools import Active
from utils.visutils import Visualizer
from utils.meta_utils import save_models
from retinanet.utils import unnormalizer
from utils.visutils import draw_line
import debugging_settings


class Training:
    def __init__(
            self,
            annotations_file_path,
            validation_file_path,
            classes_path,
            images_dir,
            epochs=50,
            use_gpu=True,
    ):
        self._classes_path = classes_path
        self._epochs = epochs
        self._img_dir = images_dir
        self._annotations_file_path = annotations_file_path
        self._device = "cuda:0" if (
            use_gpu and torch.cuda.is_available()) else "cpu"
        self._metrics = list()
        self._min_loss = inf
        self._max_mAP = 0.0
        self._val_loader = dataloader.CSVDataset(
            train_file=validation_file_path,
            class_list=self._classes_path,
            transform=transforms.Compose(
                [dataloader.Normalizer(), dataloader.Resizer()]),
            images_dir=self._img_dir,
            image_extension=".jpg",
        )

    @staticmethod
    def write_images(dataloader):
        write_dir = osp.join(
            debugging_settings.write_dir,
            "train",
            f"{debugging_settings.CYCLE_NUM:02d}",
        )
        os.makedirs(write_dir, exist_ok=True)
        print("writing annotations...")
        for i, data in enumerate(dataloader):
            im = data['img'].detach().cpu()
            if len(im.size()) == 3:
                im = im.numpy()
            else:  # len(im.size()) == 4
                im = torch.squeeze(im.permute(0, 2, 3, 1)).numpy()
            annotations = torch.squeeze(data['annot'])
            img = cv2.cvtColor(unnormalizer(image=im), cv2.COLOR_BGR2RGB)
            for annot in annotations:
                x, y, alpha = annot[0], annot[1], annot[2]
                img = draw_line(
                    img,
                    (x, y),
                    alpha,
                    line_color=(0, 0, 0),
                    center_color=(0, 0, 0),
                    half_line=True,
                    distance_thresh=40,
                    line_thickness=2,
                )
            name = data['name']
            writpe_path = osp.join(write_dir, name + '.jpg')
            cv2.imwrite(writpe_path, img)
            print("{0}/{1}".format(i + 1, len(dataloader)), end='\r')
        print("writing done.")
        return 0

    def _load_training_data(self, loader_directory=None):
        dataset_train = dataloader.CSVDataset(
            train_file=self._annotations_file_path,
            class_list=self._classes_path,
            images_dir=self._img_dir,
            transform=transforms.Compose(
                [dataloader.Normalizer(), dataloader.Augmenter(), dataloader.Resizer()]),
            save_output_img_directory=loader_directory,
        )
        sampler = dataloader.AspectRatioBasedSampler(
            dataset_train, batch_size=1, drop_last=False)
        train_loader = DataLoader(
            dataset_train, num_workers=2, collate_fn=dataloader.collater, batch_sampler=sampler)
        # self.write_images(dataloader=train_loader, image_names=dataset_train.image_names)
        return train_loader

    def _load_model(self, state_dict_path, num_classes, learning_rate=1e-5):
        retinanet = model.vgg7(num_classes=num_classes, pretrained=True)
        retinanet = retinanet.to(device=self._device)
        retinanet = torch.nn.DataParallel(retinanet)
        retinanet = retinanet.to(device=self._device)

        optimizer = torch.optim.Adam(retinanet.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True)

        checkpoint = torch.load(state_dict_path)
        retinanet.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return retinanet, optimizer, scheduler

    def _evaluate(self, model, write_parent_directory=None):
        if write_parent_directory is None:
            visualizer = None
            write_dir = None
        else:
            write_dir = osp.join(write_parent_directory, 'evaluation')
            visualizer = Visualizer()
            os.makedirs(write_dir, exist_ok=True)
        mAP = csv_eval.evaluate(self._val_loader, model,
                                visualizer=visualizer, write_dir=write_dir)
        mAP = mAP[0][0]
        return mAP

    def _save_model(self, save_model_directory, model, optimizer=None, scheduler=None, mAP=0.0, loss=inf):
        if mAP > self._max_mAP:
            save_models(
                model_path=osp.join(save_model_directory, "best_mAP_model.pt"),
                state_dict_path=osp.join(
                    save_model_directory, "best_mAP_state_dict.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            self._max_mAP = mAP
        if loss < self._min_loss:
            save_models(
                model_path=osp.join(save_model_directory,
                                    "best_loss_model.pt"),
                state_dict_path=osp.join(
                    save_model_directory, "best_loss_state_dict.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            self._min_loss = loss

    def train(self, state_dict_path, models_directory, cycle_num=-1):
        train_loader = self._load_training_data()
        # self.write_images(dataloader=train_loader)
        lr = 1e-5
        retinanet, optimizer, scheduler = self._load_model(
            state_dict_path=state_dict_path, num_classes=1, learning_rate=lr)
        self._min_loss = inf
        self._max_mAP = 0.0

        print("initial evaluation...")
        mAP = self._evaluate(
            model=retinanet,
            write_parent_directory=None,  # write_parent_directory=results_directory,
        )
        print(f"initial mAP: {mAP}")
        self._save_model(save_model_directory=models_directory, model=retinanet,
                         optimizer=optimizer, scheduler=scheduler, mAP=mAP)
        num_cuda_errors = 0
        for epoch_num in range(self._epochs):
            debugging_settings.EPOCH_NUM = epoch_num
            retinanet.train()
            retinanet.training = True
            epoch_loss = []
            for iter, data in enumerate(train_loader):
                params = [data['img'].cuda().float(), data['annot'], data['name']]
                optimizer.zero_grad()
                try:
                    losses = retinanet(params)
                except RuntimeError:
                    logging.error("cuda out of memory", exc_info=True)
                    num_cuda_errors += 1
                    continue
                classification_loss, xydistance_regression_loss, angle_distance_regression_losses = losses
                classification_loss = classification_loss.mean()
                xydistance_regression_loss = xydistance_regression_loss.mean()
                angle_distance_regression_losses = angle_distance_regression_losses.mean()
                loss = classification_loss + xydistance_regression_loss + \
                    angle_distance_regression_losses
                if loss == 0:
                    continue
                try:
                    loss.backward()
                except RuntimeError:
                    num_cuda_errors += 1
                    print("CUDA out of memory.")
                    continue
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                epoch_loss.append(loss_value)
                print(
                    'Cycle: {0:02d} | Epoch: {1:03d}  | Classification loss: {2:1.3f} | XY Regression loss: {3:1.3f} | Angle Regression loss: {4:1.3f}| Running loss: {5:1.3f}'.format(
                        cycle_num,
                        epoch_num,
                        float(classification_loss),
                        float(xydistance_regression_loss),
                        float(angle_distance_regression_losses),
                        loss_value,
                    )
                )
            mAP = self._evaluate(model=retinanet)
            mean_epoch_loss = np.mean(epoch_loss)
            print(
                "End epoch | Cycle: {0:02d} | Epoch: {1:03d} | Loss: {2:1.3f} | mAP: {3:1.3f}".format(
                    cycle_num,
                    epoch_num,
                    mean_epoch_loss,
                    mAP,
                )
            )
            metric = {'cycle': cycle_num, 'epoch': epoch_num,
                      'mAP': mAP, 'loss': mean_epoch_loss, 'lr': lr}
            self._metrics.append(metric)
            self._save_model(save_model_directory=models_directory, model=retinanet,
                             optimizer=optimizer, scheduler=scheduler, mAP=mAP, loss=mean_epoch_loss)
            scheduler.step(mean_epoch_loss)
        print(f"cuda_errors: {num_cuda_errors}")


class ActiveTraining(Training):
    def __init__(
        self,
        annotations_path,
        validation_file_path,
        oracle_annotations_path,
        classes_path,
        images_dir,
        filenames_path,
        aggregator_type,
        uncertainty_alorithm,
        budget=2,
        epochs=50,
        use_gpu=True,
        metrics_path=None,
    ):
        super().__init__(
            annotations_file_path=annotations_path,
            validation_file_path=validation_file_path,
            classes_path=classes_path,
            images_dir=images_dir,
            epochs=epochs,
            use_gpu=use_gpu,
        )
        self._image_loader = imageloader.CSVDataset(
            filenames_path=filenames_path,
            partition="unsupervised",
            class_list=self._classes_path,
            images_dir=self._img_dir,
            image_extension=".jpg",
            transform=transforms.Compose(
                [imageloader.Normalizer(), imageloader.Resizer()]),
        )
        self._gt_loader = self._load_groundtruth_data(
            gt_annotations_path=oracle_annotations_path)
        self._active = Active(
            loader=self._image_loader,
            annotations_path=annotations_path,
            class_list_path=classes_path,
            budget=budget,
            ground_truth_dataloader=self._gt_loader,
            aggregator_type=aggregator_type,
            uncertainty_algorithm=uncertainty_alorithm,
        )
        # {"num_cycle": [], "num_imgs": [], "num_labels": []}
        self._active_metrics = {"active": [], "train": self._metrics}
        self._metrics_path = metrics_path

    def _load_groundtruth_data(self, gt_annotations_path):
        gt_loader = dataloader.CSVDataset(
            train_file=gt_annotations_path,
            class_list=self._classes_path,
            images_dir=self._img_dir,
            transform=transforms.Compose(
                [dataloader.Normalizer(), dataloader.Resizer()]),
        )
        return gt_loader

    def _create_annotations(self, model):
        self._active.create_annotations(
            model=model,
        )

    def _log_active_annotations_metrics(self, num_cycle):
        if osp.isfile(self._annotations_file_path):
            f = open(self._annotations_file_path, 'r')
            reader = csv.reader(f)
            img_numbers = [int(row[0]) for row in reader]
            label_count = len(img_numbers)
            img_count = len(np.unique(img_numbers))
            f.close()
        else:
            label_count = 0
            img_count = 0
        metrics = {
            "num_cycle": num_cycle,
            "num_labels": label_count,
            "num_images": img_count,
            "image_names": img_numbers,
        }
        self._active_metrics["active"].append(metrics)
        if self._metrics_path is not None:
            with open(self._metrics_path, "w") as f:
                f.write(json.dumps(self._active_metrics))

    def run_cycle(self, cycles, init_state_dict_path, models_directory):
        for i in range(1, cycles+1):
            print(f"Cycle: {i}")
            debugging_settings.CYCLE_NUM = i
            state_dict_path = init_state_dict_path if (i == 1) else osp.join(
                models_directory, f"cycle_{i - 1:02d}", "best_mAP_state_dict.pt")
            retinanet, _, _ = self._load_model(
                state_dict_path=state_dict_path, num_classes=1)
            print("creating annotations...")
            retinanet.eval()
            retinanet.training = False
            self._create_annotations(model=retinanet)
            self.train(state_dict_path=state_dict_path, models_directory=osp.join(
                models_directory, f"cycle_{i:02d}"), cycle_num=i)
            self._log_active_annotations_metrics(num_cycle=i)
