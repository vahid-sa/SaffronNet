import os
import torch
import json
import numpy as np
from os import path as osp
from cv2 import cv2
from math import inf
from torchvision import transforms
from torch.utils.data import DataLoader
import debugging_settings
from retinanet import dataloader, csv_eval
from retinanet import my_model as model
from prediction import imageloader
from utils.active_tools import Active
from utils.visutils import draw_line, Visualizer
from utils.meta_utils import save_models
from retinanet.utils import ActiveLabelMode


class Training:
    def __init__(
            self,
            annotations_file_path,
            validation_file_path,
            classes_path,
            states_dir,
            images_dir,
            metrics_path=None,
            epochs=50,
            use_gpu=True,
    ):
        self._classes_path = classes_path
        self._epochs = epochs
        self._img_dir = images_dir
        self._states_dir = states_dir
        self._annotations_file_path = annotations_file_path
        self._device = "cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self._metrics = {'cycle': [], 'epoch':[], 'mAP': [], 'loss': [], 'lr': []}
        self._metrics_path = metrics_path
        self._min_loss = inf
        self._max_mAP = 0.0
        self._val_loader = dataloader.CSVDataset(
            train_file=validation_file_path,
            class_list=self._classes_path,
            transform=transforms.Compose([dataloader.Normalizer(), dataloader.Resizer()]),
            images_dir=self._img_dir,
            image_extension=".jpg",
        )
        os.makedirs(osp.dirname(self._metrics_path), exist_ok=True)

    def _load_training_data(self, loader_directory=None):
        dataset_train = dataloader.CSVDataset(
            train_file=self._annotations_file_path,
            class_list=self._classes_path,
            images_dir=self._img_dir,
            ground_truth_states_directory=self._states_dir,
            transform=transforms.Compose([dataloader.Normalizer(), dataloader.Augmenter(), dataloader.Resizer()]),
            save_output_img_directory=loader_directory,
        )
        sampler = dataloader.AspectRatioBasedSampler(
            dataset_train, batch_size=1, drop_last=False)
        train_loader = DataLoader(dataset_train, num_workers=2, collate_fn=dataloader.collater, batch_sampler=sampler)
        return train_loader

    def _load_model(self, state_dict_path, num_classes, learning_rate=1e-5):
        retinanet = model.vgg7(num_classes=num_classes, pretrained=True)
        retinanet = retinanet.to(device=self._device)
        retinanet = torch.nn.DataParallel(retinanet)
        retinanet = retinanet.to(device=self._device)

        optimizer = torch.optim.Adam(retinanet.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

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
        mAP = csv_eval.evaluate(self._val_loader, model, visualizer=visualizer, write_dir=write_dir)
        mAP = mAP[0][0]
        return mAP

    def _save_model(self, save_model_directory, model, optimizer=None,scheduler=None, mAP=0.0, loss=inf):
        if mAP > self._max_mAP:
            save_models(
                model_path=osp.join(save_model_directory, "best_mAP_model.pt"),
                state_dict_path=osp.join(save_model_directory, "best_mAP_state_dict.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            self._max_mAP = mAP
        if loss < self._min_loss:
            save_models(
                model_path=osp.join(save_model_directory, "best_loss_model.pt"),
                state_dict_path=osp.join(save_model_directory, "best_loss_state_dict.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            self._min_loss = loss

    def _update_metrics(self, cycle, epoch, mAP, loss, lr,):
        self._metrics['cycle'].append(cycle)
        self._metrics['epoch'].append(epoch)
        self._metrics['mAP'].append(mAP)
        self._metrics['loss'].append(loss)
        self._metrics['lr'].append(lr)
        metrics = json.dumps(self._metrics)
        with open(self._metrics_path, "w") as f:
            f.write(metrics)

    def train(self, state_dict_path, models_directory, cycle_num=-1):
        train_loader = self._load_training_data()
        retinanet, optimizer, scheduler = self._load_model(state_dict_path=state_dict_path, num_classes=1)
        self._min_loss = inf
        self._max_mAP = 0.0

        print("initial evaluation...")
        mAP = self._evaluate(
            model=retinanet,
            write_parent_directory=None,  # write_parent_directory=results_directory,
        )
        print(f"initial mAP: {mAP}")
        self._save_model(save_model_directory=models_directory, model=retinanet, optimizer=optimizer, scheduler=scheduler, mAP=mAP)

        for epoch_num in range(self._epochs):
            debugging_settings.EPOCH_NUM = epoch_num + 1
            retinanet.train()
            retinanet.training = True
            optimizer.zero_grad()
            epoch_loss = []
            for _, data in enumerate(train_loader):
                params = [data['img'].cuda().float(), data['annot'], data['gt_state'].cuda(), None, None]
                classification_loss, xydistance_regression_loss, angle_distance_regression_losses = retinanet(params)
                classification_loss = classification_loss.mean()
                xydistance_regression_loss = xydistance_regression_loss.mean()
                angle_distance_regression_losses = angle_distance_regression_losses.mean()
                loss = classification_loss + xydistance_regression_loss + angle_distance_regression_losses
                if loss == 0:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                epoch_loss.append(float(loss))
                print(
                    'Cycle: {0:02d} | Epoch: {1:03d}  | Classification loss: {2:1.3f} | XY Regression loss: {3:1.3f} | Angle Regression loss: {4:1.3f}| Running loss: {5:1.3f}'.format(
                        cycle_num,
                        epoch_num,
                        float(classification_loss),
                        float(xydistance_regression_loss),
                        float(angle_distance_regression_losses),
                        loss,
                    )
                )

                del classification_loss
                del xydistance_regression_loss
                del angle_distance_regression_losses

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
            self._save_model(save_model_directory=models_directory, model=retinanet, optimizer=optimizer, scheduler=scheduler, mAP=mAP, loss=mean_epoch_loss)
            self._update_metrics(cycle=cycle_num, epoch=epoch_num, mAP=mAP, loss=mean_epoch_loss, lr=optimizer.param_groups[0]['lr'])
            scheduler.step(mean_epoch_loss)


class ActiveTraining(Training):
    def __init__(
        self,
        active_annotations_path,
        corrected_annotations_path,
        validation_file_path,
        groundtruth_annotations_path,
        classes_path,
        states_dir,
        images_dir,
        metrics_path=None,
        supervised_annotations_path=None,
        filenames_path=None,
        epochs=50,
        radius=50,
        use_gpu=True,
    ):
        super().__init__(active_annotations_path, validation_file_path, classes_path, states_dir, images_dir, metrics_path=metrics_path, epochs=epochs, use_gpu=use_gpu)
        self._image_loader = imageloader.CSVDataset(
            filenames_path=filenames_path,
            partition="unsupervised",
            class_list=self._classes_path,
            images_dir=self._img_dir,
            image_extension=".jpg",
            transform=transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
        )
        self._active = Active(loader=self._image_loader, states_dir=self._states_dir, radius=radius, image_string_file_numbers_path=filenames_path, supervised_annotations_path=supervised_annotations_path)
        self._corrected_annotations_path = corrected_annotations_path
        self._active_annotations_path = active_annotations_path
        self._gt_loader = self._load_groundtruth_data(gt_annotations_path=groundtruth_annotations_path)

    def _load_groundtruth_data(self, gt_annotations_path, loader_directory=None):
        gt_loader = dataloader.CSVDataset(
            train_file=gt_annotations_path,
            class_list=self._classes_path,
            images_dir=self._img_dir,
            transform=transforms.Compose([dataloader.Normalizer(), dataloader.Resizer()]),
            save_output_img_directory=loader_directory,
        )
        return gt_loader

    def write_predicted_images(self, direc):
        os.makedirs(direc, exist_ok=True)
        active_states = self._active.states
        uncertain_detections = self._active.uncertain_predictions
        noisy_detections = self._active.noisy_predictions
        corrected_annotations = self._active.ground_truth_annotations
        active_annotations = self._active.active_annotations
        for i in range(len(self._image_loader.image_names)):
            image = cv2.imread(osp.join(self._image_loader.img_dir, self._image_loader.image_names[i] + self._image_loader.ext))
            my_mask = np.ones(shape=image.shape, dtype=np.float64)
            mask = active_states[i]
            my_mask[mask] *= 0.5
            image = image.astype(np.float64) * my_mask
            image = image.astype(np.uint8)
            image_noisy_detections = noisy_detections[noisy_detections[:, 0] == int(self._image_loader.image_names[i])]
            image_uncertain_detections = uncertain_detections[uncertain_detections[:, 0] == int(self._image_loader.image_names[i])]
            image_active_annotations = active_annotations[active_annotations[:, 0] == int(self._image_loader.image_names[i])]
            image_corrected_annotations = corrected_annotations[corrected_annotations[:, 0] == int(self._image_loader.image_names[i])]
            ground_truth_annotations = self._gt_loader[i]['annot'].detach().cpu().numpy()
            for annotation in ground_truth_annotations:
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
                image = draw_line(image, (x, y), alpha, line_color=(0, 255, 255), center_color=(0, 0, 0), half_line=True,
                                distance_thresh=40, line_thickness=2)
                cv2.putText(image, str(round(score, 2)), (x + 3, y + 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)

            for annot in image_corrected_annotations:
                x = int(annot[1])
                y = int(annot[2])
                alpha = annot[3]
                score = annot[5]
                image = draw_line(image, (x, y), alpha, line_color=(255, 255, 0), center_color=(0, 0, 0), half_line=True,
                                distance_thresh=40, line_thickness=2)

            for annot in image_active_annotations:
                x = int(annot[1])
                y = int(annot[2])
                alpha = annot[3]
                status = annot[-1]
                if status == ActiveLabelMode.corrected.value:
                    color = (0, 255, 0)
                elif status == ActiveLabelMode.noisy.value:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 0)
                image = draw_line(image, (x, y), alpha, line_color=color, center_color=(0, 0, 0), half_line=True, distance_thresh=40, line_thickness=2)
                cv2.imwrite(osp.join(direc, self._image_loader.image_names[i] + self._image_loader.ext), image)

    def _create_annotations(self, model, budget):
        self._active.create_active_annotations(
            model=model,
            budget=budget,
            ground_truth_loader=self._gt_loader,
            ground_truth_annotations_path=self._corrected_annotations_path,
            active_annotations_path=self._active_annotations_path,
            classes_list_path=self._classes_path,
        )

    def run_cycle(self, cycles, init_state_dict_path, models_directory, results_dir, budget=100):
        for i in range(1, cycles+1):
            debugging_settings.CYCLE_NUM = i
            print(f"Cycle: {i}")
            state_dict_path = init_state_dict_path if (i == 1) else osp.join(models_directory, f"cycle_{i - 1:02d}","best_mAP_state_dict.pt")
            retinanet, _, _ = self._load_model(state_dict_path=state_dict_path, num_classes=1)
            print("creating annotations...")
            retinanet.eval()
            retinanet.training = False
            self._create_annotations(model=retinanet, budget=budget)
            print("Writing created annotations images")
            results_directory = osp.join(results_dir, f"cycle_{i:02d}")
            debugging_settings.CLASSIFICATION_SCORES_PATH = osp.join(results_dir, "classification_scores")
            os.mkdir(results_directory)
            # self.write_predicted_images(direc=osp.join(results_directory, "create_annotations"))
            print("training...")
            self.train(state_dict_path=state_dict_path, models_directory=osp.join(models_directory, f"cycle_{i:02d}"), cycle_num=i)
