import numpy as np
import torch
import torch.nn as nn
from .settings import NUM_VARIABLES, MAX_ANOT_ANCHOR_ANGLE_DISTANCE, MAX_ANOT_ANCHOR_POSITION_DISTANCE


def prepare(a, b):
    # extend as cols
    repetitions = b.shape[0]
    at = torch.tile(a, (repetitions, 1))
    at = at.transpose(-1, 0)

    # extend as rows
    # bt = np.tile(b, (repetitions, 1))
    repetitions = a.shape[0]
    bt = torch.tile(b, (repetitions, 1))
    return at, bt


def distance(ax, bx):
    """
    ax: (N) ndarray of float
    bx: (K) ndarray of float
    Returns
    -------
    (N, K) ndarray of distance between all x in ax, bx
    """
    ax, bx = prepare(ax, bx)
    return torch.abs(ax - bx)


def calc_distance(a, b):
    ax = a[:, 0]
    bx = b[:, 0]

    ay = a[:, 1]
    by = b[:, 1]

    aa = a[:, 2]
    ba = b[:, 2]

    dalpha = distance(ax=aa, bx=ba)
    dx = distance(ax=ax, bx=bx)
    dy = distance(ax=ay, bx=by)
    dxy = torch.sqrt(dx*dx + dy*dy)

    return dxy,  dalpha


class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        xydistance_regression_losses = []
        angle_distance_regression_losses = []

        anchor = anchors[0, :, :]

        anchor_ctr_x = anchor[:, 0]
        anchor_ctr_y = anchor[:, 1]
        anchor_alpha = anchor[:, 2]

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            center_alpha_annotation = annotations[j, :, :]
            center_alpha_annotation = center_alpha_annotation[
                center_alpha_annotation[:, NUM_VARIABLES] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

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

            dxy, dalpha = calc_distance(
                anchors[0, :, :], center_alpha_annotation[:, :NUM_VARIABLES])

            dxy_min, dxy_argmin = torch.min(dxy, dim=1)  # num_anchors x 1

            # compute the loss for classification
            # print('classification.shape: ', classification.shape)
            # print('anchors.shape: ', anchors.shape)
            targets = torch.ones(classification.shape) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()
    # -----------------------------------------------------------------------

            targets[torch.ge(
                dxy_min, 1.5 * MAX_ANOT_ANCHOR_POSITION_DISTANCE), :] = 0

            a = dalpha[range(dalpha.shape[0]), dxy_argmin]
            targets[torch.ge(
                a, 1.5 * MAX_ANOT_ANCHOR_ANGLE_DISTANCE), :] = 0

            positive_indices = torch.logical_and(
                torch.le(
                    dxy_min, MAX_ANOT_ANCHOR_POSITION_DISTANCE),
                torch.le(
                    a, MAX_ANOT_ANCHOR_ANGLE_DISTANCE
                ))

            d_argmin = positive_indices.nonzero(as_tuple=True)[0]
            d_argmin = dxy_argmin[d_argmin]
            num_positive_anchors = positive_indices.sum()

            # assigned_annotations = center_alpha_annotation[deltaphi_argmin, :] # no different in result
            assigned_annotations = center_alpha_annotation[d_argmin, :]
            targets[positive_indices, :] = 0
            targets[positive_indices,
                    assigned_annotations[:, 3].long()] = 1

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

            classification_losses.append(
                cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[d_argmin, :]

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
                    targets = targets / \
                        torch.Tensor([[1, 1, 1]]).cuda()
                else:
                    targets = targets/torch.Tensor([[1, 1, 1]])

                regression_diff_xy = torch.abs(
                    targets[:, :2] - regression[positive_indices, :2])

                regression_diff_angle = 1 - torch.cos(
                    targets[:, 2] - regression[positive_indices, 2])

                regression_loss_xy = torch.where(
                    torch.le(regression_diff_xy, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff_xy, 2),
                    regression_diff_xy - 0.5 / 9.0)
                xydistance_regression_losses.append(regression_loss_xy.mean())
                angle_distance_regression_losses.append(
                    regression_diff_angle.mean())
            else:
                print("-----------------1")
                if torch.cuda.is_available():
                    xydistance_regression_losses.append(
                        torch.tensor(0).float().cuda())
                    angle_distance_regression_losses.append(
                        torch.tensor(0).float().cuda())
                    print("-----------------2")
                else:
                    print("-----------------3")
                    xydistance_regression_losses.append(
                        torch.tensor(0).float())
                    angle_distance_regression_losses.append(
                        torch.tensor(0).float())
                    print("-----------------4")

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
            torch.stack(xydistance_regression_losses).mean(dim=0, keepdim=True), \
            torch.stack(angle_distance_regression_losses).mean(
                dim=0, keepdim=True)
