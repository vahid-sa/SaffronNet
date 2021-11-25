from math import inf
import sys
import os
import numpy as np
from cv2 import cv2
import torch
import shutil
import logging
import json
from os import path as osp
from torch._C import import_ir_module
from torchvision import transforms
from torch.utils.data import DataLoader
import retinanet
import debugging_settings
from retinanet import model
from retinanet import dataloader, csv_eval
from utils.visutils import draw_line, Visualizer
from prediction import imageloader
from utils.active_tools import Active
from utils.meta_utils import save_models
from retinanet.utils import ActiveLabelMode


logging.basicConfig(level=logging.DEBUG)
retinanet.settings.NUM_QUERIES = 100
retinanet.settings.NOISY_THRESH = 0.5
# retinanet.settings.DAMPENING_PARAMETER = 0.0


class parser:
    ground_truth_annotations = osp.abspath("annotations/unsupervised.csv")
    csv_classes = osp.abspath("annotations/labels.csv")
    images_dir = osp.expanduser("~/Saffron/dataset/Train/")
    corrected_annotations = osp.expanduser("~/Saffron/active_annotations/corrected.csv")
    filenames = osp.abspath("annotations/filenames.json")
    partition = "unsupervised"
    ext = ".jpg"
    model_path = osp.expanduser('~/Saffron/weights/supervised/init_model.pt')
    state_dict_path = osp.expanduser('~/Saffron/init_fully_trained_weights/init_state_dict.pt')
    states_dir = osp.expanduser("~/Saffron/active_annotations/states")
    active_annotations = osp.expanduser("~/Saffron/active_annotations/train.csv")
    save_directory = osp.expanduser("~/tmp/saffron_imgs/")
    epochs = 2
    csv_val = osp.abspath("./annotations/validation.csv")
    save_models_directory = osp.expanduser("~/Saffron/weights/active")
    cycles = 3
    budget = 100
    supervised_annotations = osp.abspath("./annotations/supervised.csv")

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
        if osp.isdir(parser.save_models_directory):
            shutil.rmtree(parser.save_models_directory)
        os.makedirs(parser.save_models_directory, exist_ok=False)


class Training:
    def __init__(
            self,
            image_loader,
            gt_loader,
            val_loader,
            states_dir,
            images_dir,
            corrected_annotations_path,
            active_annotations_path,
            classes_path,
            supervised_annotations_path=None,
            filenames_path=None,
            budget=100,
            radius=50,
            epochs=10,
            use_gpu=True,
    ):
        self._gt_loader = gt_loader
        self._val_loader = val_loader
        self._corrected_annotations_path = corrected_annotations_path
        self._active_annotations_path = active_annotations_path
        self._classes_path = classes_path
        self._budget = budget
        self._epochs = epochs
        self._img_dir = images_dir
        self._states_dir = states_dir
        self._image_loader = image_loader
        self._active = Active(loader=self._image_loader, states_dir=self._states_dir, radius=radius, image_string_file_numbers_path=filenames_path, supervised_annotations_path=supervised_annotations_path)
        # self._active = Active(loader=self._image_loader, states_dir=self._states_dir, radius=radius, image_string_file_numbers_path=None, supervised_annotations_path=None)
        self._device = "cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self._metrics = {'cycle': [], 'epoch':[], 'mAP': [], 'loss': [], 'lr': []}

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
                image = draw_line(image, (x, y), alpha, line_color=color, center_color=(0, 0, 0), half_line=True,
                                distance_thresh=40, line_thickness=2)
                cv2.imwrite(osp.join(direc, self._image_loader.image_names[i] + self._image_loader.ext), image)

    def create_annotations(self, model):
        self._active.create_active_annotations(
            model=model,
            budget=self._budget,
            ground_truth_loader=self._gt_loader,
            ground_truth_annotations_path=self._corrected_annotations_path,
            active_annotations_path=self._active_annotations_path,
            classes_list_path=self._classes_path,
        )

    def _load_training_data(self, loader_directory):
        dataset_train = dataloader.CSVDataset(
            train_file=self._active_annotations_path,
            class_list=self._classes_path,
            images_dir=self._img_dir,
            ground_truth_states_directory=self._states_dir,
            transform=transforms.Compose([dataloader.Normalizer(), dataloader.Augmenter(), dataloader.Resizer()]),
            save_output_img_directory=loader_directory,
        )

        sampler = dataloader.AspectRatioBasedSampler(
            dataset_train, batch_size=1, drop_last=False)
        train_loader = DataLoader(
            dataset_train, num_workers=2, collate_fn=dataloader.collater, batch_sampler=sampler)
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

    def train(self, state_dict_path, current_cycle, models_directory=None, results_directory=None):
        train_loader = self._load_training_data(loader_directory=results_directory)
        retinanet, optimizer, scheduler = self._load_model(state_dict_path=state_dict_path, num_classes=1)

        print("initial evaluation...")
        if results_directory is None:
            visualizer = None
            write_dir = None
        else:
            visualizer = Visualizer()
            write_dir = osp.join(results_directory, "evaluation")
            os.makedirs(write_dir, exist_ok=True)
        mAP = csv_eval.evaluate(self._val_loader, retinanet, visualizer=visualizer, write_dir=write_dir)
        mAP = mAP[0][0]
        print(f"initial mAP: {mAP}")
        if models_directory is not None:
            save_models(
                model_path=osp.join(models_directory, "best_loss_model.pt"),
                state_dict_path=osp.join(models_directory, "best_loss_state_dict.pt"),
                model=retinanet,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        max_mAP = mAP
        if models_directory is not None:
            save_models(
                model_path=osp.join(models_directory, "best_mAP_model.pt"),
                state_dict_path=osp.join(models_directory, "best_mAP_state_dict.pt"),
                model=retinanet,
                optimizer=optimizer,
                scheduler=scheduler,
            )

        min_loss = inf
        max_mAP = 0.0

        for epoch_num in range(self._epochs):
            debugging_settings.EPOCH_NUM = epoch_num + 1
            retinanet.train()
            retinanet.training = True
            optimizer.zero_grad()
            loss_hist = []
            epoch_loss = []
            epoch_CLASSIFICATION_loss = []
            epoch_XY_REG_loss = []
            epoch_ANGLE_REG_loss = []
            for i, data in enumerate(train_loader):
                """ save images for models
                if results_directory is None:
                    results_directory_in_model = None
                else:
                    results_directory_in_model = osp.join(results_directory, f"epoch_{i:03d}", "model")
                    os.makedirs(results_directory_in_model, exist_ok=True)
                params = [data['img'].cuda().float(), data['annot'], data['gt_state'].cuda(), data["aug_img_path"], results_directory_in_model]
                """
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
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                epoch_CLASSIFICATION_loss.append(float(classification_loss))
                epoch_XY_REG_loss.append(float(xydistance_regression_loss))
                epoch_ANGLE_REG_loss.append(float(angle_distance_regression_losses))
                print(
                    'Classification loss: {:1.5f} | XY Regression loss: {:1.5f} | Angle Regression loss: {:1.5f}| Running loss: {:1.5f}'.format(
                        float(classification_loss),
                        float(xydistance_regression_loss),
                        float(angle_distance_regression_losses),
                        loss,
                    )
                )

                # classification_loss = classification_loss.mean()
                # xydistance_regression_loss = xydistance_regression_loss.mean()
                # angle_distance_regression_losses = angle_distance_regression_losses.mean()
                # loss = classification_loss + xydistance_regression_loss + angle_distance_regression_losses

                del classification_loss
                del xydistance_regression_loss
                del angle_distance_regression_losses

            if results_directory is None:
                visualizer = None
                write_dir = None
            else:
                visualizer = Visualizer()
                write_dir = osp.join(results_directory, "evaluation")
                os.makedirs(write_dir, exist_ok=True)
            mAP = csv_eval.evaluate(self._val_loader, retinanet, visualizer=visualizer, write_dir=write_dir)
            mAP = mAP[0][0]
            mean_epoch_loss = np.mean(epoch_loss)
            print(f"epoch loss: {mean_epoch_loss}, epoch mAP: {mAP}")
            self._metrics['cycle'].append(current_cycle)
            self._metrics['epoch'].append(epoch_num)
            self._metrics['mAP'].append(mAP)
            self._metrics['loss'].append(mean_epoch_loss)
            self._metrics['lr'].append(optimizer.param_groups[0]['lr'])
            metrics = json.dumps(self._metrics)
            with open(osp.join(osp.dirname(results_directory), "metrics.json"), "w") as f:
                f.write(metrics)
            if mean_epoch_loss < min_loss:
                min_loss = mean_epoch_loss
                print("Minimum loss")
                if models_directory is not None:
                    save_models(
                        model_path=osp.join(models_directory, "best_loss_model.pt"),
                        state_dict_path=osp.join(models_directory, "best_loss_state_dict.pt"),
                        model=retinanet,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
            if mAP > max_mAP:
                max_mAP = mAP
                print("Minimum mAP")
                if models_directory is not None:
                    save_models(
                        model_path=osp.join(models_directory, "best_mAP_model.pt"),
                        state_dict_path=osp.join(models_directory, "best_mAP_state_dict.pt"),
                        model=retinanet,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
            scheduler.step(mean_epoch_loss)

    def run_cycle(self, cycles, init_state_dict_path, models_directory, results_dir):
        for i in range(1, cycles+1):
            debugging_settings.CYCLE_NUM = i
            print(f"Cycle: {i}")
            state_dict_path = init_state_dict_path if (i == 1) else osp.join(models_directory, f"cycle_{i - 1:02d}","best_mAP_state_dict.pt")
            retinanet, _, _ = self._load_model(state_dict_path=state_dict_path, num_classes=1)
            print("creating annotations...")
            retinanet.eval()
            retinanet.training = False
            self.create_annotations(model=retinanet)
            print("Writing created annotations images")
            results_directory = osp.join(results_dir, f"cycle_{i:02d}")
            debugging_settings.CLASSIFICATION_SCORES_PATH = osp.join(results_dir, "classification_scores")
            os.mkdir(results_directory)
            # self.write_predicted_images(direc=osp.join(results_directory, "create_annotations"))
            print("training...")
            self.train(state_dict_path=state_dict_path, models_directory=osp.join(models_directory, f"cycle_{i:02d}"), results_directory=results_directory, current_cycle=i)


def main():
    parser.reset()

    image_loader = imageloader.CSVDataset(
        filenames_path=parser.filenames,
        partition=parser.partition,
        class_list=parser.csv_classes,
        images_dir=parser.images_dir,
        image_extension=parser.ext,
        transform=transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
    )

    gt_loader = dataloader.CSVDataset(
        train_file=parser.ground_truth_annotations,
        class_list=parser.csv_classes,
        images_dir=parser.images_dir,
        transform=transforms.Compose([dataloader.Normalizer(), dataloader.Resizer()]),
        save_output_img_directory=parser.save_directory,
    )

    val_loader = dataloader.CSVDataset(
        train_file=parser.csv_val,
        class_list=parser.csv_classes,
        transform=transforms.Compose([dataloader.Normalizer(), dataloader.Resizer()]),
        images_dir=parser.images_dir,
        image_extension=parser.ext,
    )

    states_dir = parser.states_dir
    images_dir = parser.images_dir

    trainer = Training(
        image_loader=image_loader,
        gt_loader=gt_loader,
        val_loader=val_loader,
        states_dir=states_dir,
        images_dir=images_dir,
        corrected_annotations_path=parser.corrected_annotations,
        active_annotations_path=parser.active_annotations,
        classes_path=parser.csv_classes,
        epochs=parser.epochs,
        budget=parser.budget,
        supervised_annotations_path=parser.supervised_annotations,
        filenames_path=parser.filenames,
    )
    trainer.run_cycle(cycles=parser.cycles, init_state_dict_path=parser.state_dict_path, models_directory=parser.save_models_directory, results_dir=parser.save_directory)


if __name__ == "__main__":
    main()
