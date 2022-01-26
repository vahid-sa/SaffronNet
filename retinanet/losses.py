import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import gc
from os import path as osp
from .utils import calc_distance
from .settings import NUM_VARIABLES, MAX_ANOT_ANCHOR_ANGLE_DISTANCE, MAX_ANOT_ANCHOR_POSITION_DISTANCE
import retinanet
from utils.visutils import draw_line


class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations, states):
        alpha = 0.95
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        xydistance_regression_losses = []
        angle_distance_regression_losses = []
        anchor = anchors[0, :, :]

        anchor_ctr_x = anchor[:, 0]
        anchor_ctr_y = anchor[:, 1]
        anchor_alpha = anchor[:, 2]
        # print(f"x: {anchor_ctr_x.max()}\ny: {anchor_ctr_y.max()}\na: {anchor_alpha.max()}")
        for j in range(batch_size):
            classification = classifications[j, :, :]
            # with open(debugging_settings.CLASSIFICATION_SCORES_PATH, "a") as f:
            #     hist, _ = np.histogram(classification.detach().cpu().numpy(), bins=10, range=(0.0, 1.0))
            #     f.write(json.dumps({"cycle": debugging_settings.CYCLE_NUM, "epoch": debugging_settings.EPOCH_NUM, "scores": hist.tolist()}))
            regression = regressions[j, :, :]
            center_alpha_annotation = annotations[j, :, :]
            center_alpha_annotation = center_alpha_annotation[
                center_alpha_annotation[:, NUM_VARIABLES] != -1]
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            state = states[j, :, :, :]
            gt_map = FocalLoss.set_noisy_anchors(state=state, anchor=anchor)
            # img = torch.zeros(size=(state.shape[:2]), dtype=torch.uint8)
            true_indices = anchor[gt_map].to(dtype=torch.int64, device='cpu')
            true_indices = true_indices[:, [1, 0]]
            true_indices = true_indices.t()
            true_indices = tuple(true_indices.tolist())
            # img[true_indices] = 255
            # img = img.numpy()
            if center_alpha_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(
                        classification.shape).cuda() * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * \
                                   torch.pow(focal_weight, gamma)
                    bce = -(torch.log(1.0 - classification))
                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    xydistance_regression_losses.append(
                        torch.tensor(0).float().cuda())
                    angle_distance_regression_losses.append(
                        torch.tensor(0).float().cuda())
                else:
                    alpha_factor = torch.ones(
                        classification.shape) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * \
                                   torch.pow(focal_weight, gamma)
                    bce = -(torch.log(1.0 - classification))
                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    xydistance_regression_losses.append(
                        torch.tensor(0).float())
                    angle_distance_regression_losses.append(
                        torch.tensor(0).float())
                continue
            anchors_for_calc_distance = anchors[0, :, :]
            center_alpha_annotation_for_calc_distance = center_alpha_annotation[:, :NUM_VARIABLES]
            dxy, dalpha = calc_distance(
                anchors_for_calc_distance, center_alpha_annotation_for_calc_distance)
            dxy_min, dxy_argmin = torch.min(dxy, dim=1)  # num_anchors x 1
            a = dalpha[range(dalpha.shape[0]), dxy_argmin]
            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()
            # -----------------------------------------------------------------------
            targets[torch.ge(
                dxy_min, 1.5 * MAX_ANOT_ANCHOR_POSITION_DISTANCE), :] = 0
            targets[torch.ge(
                a, 1.5 * MAX_ANOT_ANCHOR_ANGLE_DISTANCE), :] = 0
            positive_indices = FocalLoss.get_positive_indices(min_distances=dxy_min, accepted_dalpha=a)
            assigned_annotations = FocalLoss.get_assigned_annotations(
                annotation=center_alpha_annotation,
                min_distances_args=dxy_argmin,
                positive_indices=positive_indices,
            )
            num_positive_anchors = positive_indices.sum()
            targets[positive_indices, :] = 0
            # change for ground_truth background
            # assert sum(positive_indices) == assigned_annotations.shape[0], "only one index for each annotation"
            targets[positive_indices,
                    assigned_annotations[:, 3].long()] = 1
            # -------------------------------------------------------------------------
            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha
            alpha_factor = torch.where(
                torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(
                torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) +
                    (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce
            if torch.cuda.is_available():
                cls_loss = torch.where(
                    torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(
                    torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))
            dampening_factor = torch.where(gt_map, 1.0, retinanet.settings.DAMPENING_PARAMETER)
            cls_loss *= torch.unsqueeze(dampening_factor, dim=1)
            # self.write_loss(states=state, anchors_like=anchor[:, :2], cls_losses=cls_loss)
            # logging.debug(f"dampening_parameter: {retinanet.settings.DAMPENING_PARAMETER}")
            # cv2.imwrite(osp.expanduser(f'~/tmp/{np.random.randint(1000)}.jpg'), img)
            classification_losses.append(
                cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
            # compute the loss for regression
            if positive_indices.sum() > 0:
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                anchor_alpha_pi = anchor_alpha[positive_indices]
                gt_ctr_x = assigned_annotations[:, 0]
                gt_ctr_y = assigned_annotations[:, 1]
                gt_alpha = assigned_annotations[:, 2]

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi)
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi)
                targets_dalpha = (gt_alpha - anchor_alpha_pi)

                targets = torch.stack(
                    (targets_dx, targets_dy, targets_dalpha))
                targets = targets.t()
                if torch.cuda.is_available():
                    targets = targets.cuda()
                    targets = targets / torch.Tensor([[1, 1, 1]]).cuda()
                else:
                    targets = targets/torch.Tensor([[1, 1, 1]])

                regression_diff_xy = torch.abs(
                    targets[:, :2] - regression[positive_indices, :2])

                regression_diff_angle = (torch.abs(
                    targets[:, 2] - regression[positive_indices, 2]) - 10) / 5
                # 5 degree mismatch is normal in annotations
                if torch.cuda.is_available():
                    regression_diff_angle = torch.where(
                        torch.le(regression_diff_angle, 0), torch.zeros(cls_loss.shape).cuda(), regression_diff_angle)
                else:
                    regression_diff_angle = torch.where(
                        torch.le(regression_diff_angle, 0), torch.zeros(cls_loss.shape), regression_diff_angle)

                regression_loss_xy = torch.where(
                    torch.le(regression_diff_xy, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff_xy, 2),
                    regression_diff_xy - 0.5 / 9.0)
                regression_loss_xy *= torch.unsqueeze(dampening_factor[positive_indices], dim=1)
                regression_diff_angle *= dampening_factor[positive_indices]
                xydistance_regression_losses.append(regression_loss_xy.mean())
                angle_distance_regression_losses.append(
                    regression_diff_angle.mean())
            else:
                if torch.cuda.is_available():
                    xydistance_regression_losses.append(
                        torch.tensor(0).float().cuda())
                    angle_distance_regression_losses.append(
                        torch.tensor(0).float().cuda())
                else:
                    xydistance_regression_losses.append(
                        torch.tensor(0).float())
                    angle_distance_regression_losses.append(
                        torch.tensor(0).float())
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(xydistance_regression_losses).mean(dim=0, keepdim=True), \
               torch.stack(angle_distance_regression_losses).mean(
                   dim=0, keepdim=True)

    @staticmethod
    def get_positive_indices(min_distances, accepted_dalpha):
        positive_indices = torch.logical_and(
            torch.le(min_distances, MAX_ANOT_ANCHOR_POSITION_DISTANCE),
            torch.le(accepted_dalpha, MAX_ANOT_ANCHOR_ANGLE_DISTANCE),
        )
        return positive_indices

    @staticmethod
    def get_assigned_annotations(annotation, positive_indices, min_distances_args):
        d_argmin = min_distances_args[positive_indices.nonzero(as_tuple=True)[0]]
        assigned_annotations = annotation[d_argmin, :]
        return assigned_annotations

    @staticmethod
    def get_dampening_factor(annotations, targets, positive_indices, background_positive_indices, min_distances_args):
        DAMPENING_PARAMETER = retinanet.settings.DAMPENING_PARAMETER
        targets_max, _ = targets.max(axis=1)
        ignored_background = torch.logical_or((targets_max == -1), (targets_max == 0))
        assert torch.logical_and(ignored_background, positive_indices).sum() == 0.0, "Overlap between positive indices and non positive indices!!!"
        assert torch.logical_or(ignored_background,
                                positive_indices).sum() == positive_indices.shape[0], "Some indices not in background, ignored or positive!!!"
        assert torch.logical_and(positive_indices, background_positive_indices).sum() == 0.0

        dampening_factor = torch.full(size=(targets.shape[0],), dtype=torch.float64, fill_value=DAMPENING_PARAMETER)
        if torch.cuda.is_available():
            dampening_factor = dampening_factor.cuda()
        dampening_factor[targets_max == -1] = 1.0
        dampening_factor[targets_max == 0] = DAMPENING_PARAMETER
        dampening_factor[background_positive_indices] = 1.0
        accepted_annotations_indices = min_distances_args[positive_indices]
        accepted_annotations_status = torch.squeeze(annotations[:, accepted_annotations_indices, -1])
        dampening_factor[positive_indices] = torch.where(
            accepted_annotations_status == 1.0, 1.0, DAMPENING_PARAMETER).type(dampening_factor.dtype)
        return dampening_factor

    @staticmethod
    def set_noisy_anchors(anchor, state):
        state = torch.squeeze(state)
        points = tuple(torch.round(anchor[:, [1, 0]]).type(torch.LongTensor).detach().cpu().numpy().T.tolist())
        return state[points]

    @staticmethod
    def write_loss(states: torch.Tensor, anchors_like: torch.Tensor, cls_losses: torch.Tensor, write_dir: str = "~/st/Saffron/tmp/"):
        write_dir = osp.abspath(osp.expanduser(osp.expanduser(write_dir)))
        canvas = np.zeros(shape=states.size(), dtype=np.uint8)
        points = tuple(torch.round(anchors_like[:, [1, 0]]).type(torch.LongTensor).detach().cpu().numpy().T.tolist())
        values = np.where(cls_losses.detach().cpu().numpy() !=0, 255, 0) 
        canvas[points] = values
        os.makedirs(write_dir, exist_ok=True)
        name = f"{len(os.listdir(write_dir)) + 1:03d}.png"
        path = osp.join(write_dir, name)
        cv2.imwrite(path, canvas)
        return 0

    @staticmethod
    def write_img_loss(read_path, cls_loss, write_dir, anchors_like):
        img = cv2.imread(read_path)
        img = np.full(shape=img.shape, fill_value=255.0, dtype=np.float64)
        img_channels = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        cls_loss = np.squeeze(cls_loss.detach().cpu().numpy()).astype(np.float64)
        cls_loss = (cls_loss - np.min(cls_loss)) / (np.max(cls_loss) - np.min(cls_loss))
        points = tuple(torch.round(anchors_like[:, [1, 0]]).type(torch.LongTensor).detach().cpu().numpy().T.tolist())
        for i in range(len(img_channels)):
            img_channels[i][points] *= cls_loss
        img = np.concatenate([img_channels[0][:, :, np.newaxis], img_channels[1][:, :, np.newaxis], img_channels[2][:, :, np.newaxis]], axis=-1)
        img = img.astype(np.uint8)
        write_path = osp.join(write_dir, osp.basename(read_path))
        cv2.imwrite(write_path, img)

    @staticmethod
    def write_positive_indices(regressions, anchors, annotations, positive_indices, active_states, read_path, write_dir):
        predictions = regressions + anchors
        predictions = predictions[positive_indices]
        img = cv2.imread(read_path)
        img = img.astype(np.float64)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        active_states = np.squeeze(active_states.cpu().detach().numpy())
        b[active_states] *= 0.5
        g[active_states] *= 0.5
        r[active_states] *= 0.5
        img = img.astype(np.uint8)
        for prediction in predictions:
            x, y, alpha = prediction
            img = draw_line(img, (x, y), alpha, line_color=(255, 255, 0), center_color=(0, 0, 0), half_line=True,
                              distance_thresh=40, line_thickness=2)
        for annotation in annotations:
            x, y, alpha = annotation[:3]
            status = annotation[-1]
            if status == 1:
                color = (0, 255, 0)
            elif status == 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 255)
            img = draw_line(img, (x, y), alpha, line_color=color, center_color=(0, 0, 0), half_line=True,
                            distance_thresh=40, line_thickness=2)
        write_path = osp.join(write_dir, osp.basename(read_path))
        cv2.imwrite(write_path, img)
