import torch
import torch.nn as nn
import gc
from retinanet.settings import NUM_VARIABLES, MAX_ANOT_ANCHOR_ANGLE_DISTANCE, MAX_ANOT_ANCHOR_POSITION_DISTANCE
import retinanet
# from visualize_anchors import visualize_anchors


def absolute(tensor: torch.tensor, large_matrix: bool):
    """
    if large_matrix:
        # print("large matrix shape: {0}".format(tensor.shape))
        # denominator = 5
        # for i in range(int(tensor.shape[0] / denominator)):
        #     gc.collect()
        #     lower_range = i * denominator
        #     upper_range = (i + 1) * denominator
        #     tensor[lower_range:upper_range, :] = torch.abs(tensor[lower_range:upper_range, :])
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                tensor[i, j] = (tensor[i, j] ** 2) ** 0.5
    else:
        row_index = int(tensor.shape[0] / 2)
        col_index = int(tensor.shape[1] / 2)
        tensor[:row_index, :col_index] = torch.abs(tensor[:row_index, :col_index])
        tensor[:row_index, col_index:] = torch.abs(tensor[:row_index, col_index:])
        tensor[row_index:, :col_index] = torch.abs(tensor[row_index:, :col_index])
        tensor[row_index:, col_index:] = torch.abs(tensor[row_index:, col_index:])
    """
    # if large_matrix:
    # print("large matrix:", tensor.shape)
    #    # for i in range(tensor.shape[0]):
    #    #     for j in range(tensor.shape[1]):
    #    #         tensor[i, j] = tensor[i, j] ** 2
    #    #         tensor[i, j] = tensor[i, j] ** 0.5
    row_index = int(tensor.shape[0] / 2)
    col_index = int(tensor.shape[1] / 2)
    tensor[:row_index, :col_index] = torch.abs(tensor[:row_index, :col_index])
    tensor[:row_index, col_index:] = torch.abs(tensor[:row_index, col_index:])
    tensor[row_index:, :col_index] = torch.abs(tensor[row_index:, :col_index])
    tensor[row_index:, col_index:] = torch.abs(tensor[row_index:, col_index:])


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


def distance(ax, bx, large_matrix: bool = False):
    """
    ax: (N) ndarray of float
    bx: (K) ndarray of float
    Returns
    -------
    (N, K) ndarray of distance between all x in ax, bx
    """
    gc.collect()
    torch.cuda.empty_cache()
    ax_prepared, bx_prepared = prepare(ax, bx)

    gc.collect()
    torch.cuda.empty_cache()

    dist = ax_prepared - bx_prepared
    del ax_prepared, bx_prepared
    gc.collect()
    torch.cuda.empty_cache()
    absolute(tensor=dist, large_matrix=large_matrix)
    return dist


def calc_distance(a, b):
    ax = a[:, 0]
    bx = b[:, 0]
    dx = distance(ax=ax, bx=bx)
    del ax, bx

    ay = a[:, 1]
    by = b[:, 1]
    dy = distance(ax=ay, bx=by)
    del ay, by

    aa = a[:, 2]
    ba = b[:, 2]
    dalpha = distance(ax=aa, bx=ba)
    del aa, ba

    dxy = torch.sqrt(dx * dx + dy * dy)
    del dx, dy

    return dxy, dalpha


class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
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
        for j in range(batch_size):

            gc.collect()
            torch.cuda.empty_cache()
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
            gc.collect()
            torch.cuda.empty_cache()
            anchors_for_calc_distance = anchors[0, :, :].cpu()
            center_alpha_annotation_for_calc_distance = center_alpha_annotation[:, :NUM_VARIABLES].cpu()
            dxy, dalpha = calc_distance(
                anchors_for_calc_distance, center_alpha_annotation_for_calc_distance)
            dxy_min, dxy_argmin = torch.min(dxy, dim=1)  # num_anchors x 1
            a = dalpha[range(dalpha.shape[0]), dxy_argmin]
            if torch.cuda.is_available():
                dxy_min = dxy_min.cuda()
                dxy_argmin = dxy_argmin.cuda()
                a = a.cuda()
            del anchors_for_calc_distance, center_alpha_annotation_for_calc_distance, dxy, dalpha

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()
            # -----------------------------------------------------------------------
            targets[torch.ge(
                dxy_min, 1.5 * MAX_ANOT_ANCHOR_POSITION_DISTANCE), :] = 0
            targets[torch.ge(
                a, 1.5 * MAX_ANOT_ANCHOR_ANGLE_DISTANCE), :] = 0

            positive_indices, background_positive_indices = FocalLoss.get_positive_indices(
                annotation=center_alpha_annotation,
                min_distances=dxy_min,
                min_distances_args=dxy_argmin,
                accepted_dalpha=a,
            )
            assigned_annotations = FocalLoss.get_assigned_annotations(
                annotation=center_alpha_annotation,
                min_distances_args=dxy_argmin,
                positive_indices=positive_indices,
            )
            num_positive_anchors = positive_indices.sum()
            targets[positive_indices, :] = 0
            # change for ground_truth background
            assert sum(positive_indices) == assigned_annotations.shape[0], "only one index for each annotation"

            targets[positive_indices,
                    assigned_annotations[:, 3].long()] = 1

            # -------------------------------------------------------------------------
            """
            dampening_factor = FocalLoss.get_dampening_factor(
                annotations=annotations,
                targets=targets,
                positive_indices=positive_indices,
                background_positive_indices=background_positive_indices,
                min_distances_args=dxy_argmin,
            )
            """
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
            # cls_loss *= torch.unsqueeze(dampening_factor, dim=1)

            classification_losses.append(
                cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
            # if retinanet.settings.save_image_dir is not None:
            #     visualize_anchors(
            #         anchors=anchors,
            #         targets=targets,
            #         annots=center_alpha_annotation,
            #         save_image_dir=retinanet.settings.save_image_dir,
            #         load_image_path=load_image_paths[j],
            #     )
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
                    targets = targets / torch.Tensor([[1, 1, 1]])

                negative_indices = 1 + (~positive_indices)
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
                # regression_loss_xy *= torch.unsqueeze(dampening_factor[positive_indices], dim=1)
                # regression_diff_angle *= dampening_factor[positive_indices]
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
    def get_positive_indices(annotation, min_distances, min_distances_args, accepted_dalpha):
        positive_indices = torch.logical_and(
            torch.le(min_distances, MAX_ANOT_ANCHOR_POSITION_DISTANCE),
            torch.le(accepted_dalpha, MAX_ANOT_ANCHOR_ANGLE_DISTANCE),
        )
        assigned_labels = annotation[min_distances_args, 3]
        background_positive_indices = torch.logical_and(positive_indices, torch.eq(assigned_labels, -1))
        foreground_positive_indices = torch.logical_and(positive_indices, torch.ne(assigned_labels, -1))
        return foreground_positive_indices, background_positive_indices

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
        assert torch.logical_and(ignored_background,
                                 positive_indices).sum() == 0.0, "Overlap between positive indices and non positive indices!!!"
        assert torch.logical_or(ignored_background,
                                positive_indices).sum() == positive_indices.shape[
                   0], "Some indices not in background, ignored or positive!!!"
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

