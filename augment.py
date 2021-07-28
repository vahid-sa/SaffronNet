import os
import numpy as np
import torch
import torchvision
import cv2 as cv
import shutil
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import csv
import random
import argparse
from os import path as osp
import copy
from utils.visutils import DrawMode, get_alpha, get_dots, std_draw_line, draw_line
from prediction import imageloader

NUM_VARIABLES = 3
aug_detection_number = 0


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    """ #
    """

    def __init__(self) -> None:
        super().__init__()
        ia.seed(3)
        self.seq = iaa.Sequential([
            # iaa.Affine(
            #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            #     rotate=(-15, 15),
            #     shear=(-4, 4)
            # ),
            iaa.Fliplr(1.0),  # horizontal flips
            # color jitter, only affects the image
            # iaa.AddToHueAndSaturation((-50, 50))
        ])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        print("orig_image:", image.shape, image.dtype)
        new_annots = annots.copy()
        kps = []
        # Alphas = list()
        for x, y, alpha in annots[:, 1:NUM_VARIABLES+1]:
            x0, y0, x1, y1 = get_dots(x, y, alpha, distance_thresh=60)
            kps.append(Keypoint(x=x0, y=y0))
            kps.append(Keypoint(x=x1, y=y1))
            # Alphas.append(alpha)
            # Alphas.append(alpha)
            # cv.circle(sample["img"], (x1, y1), 1, (0, 255, 0))

        kpsoi = KeypointsOnImage(kps, shape=image.shape)

        image_aug, kpsoi_aug = self.seq(image=image, keypoints=kpsoi)
        print("image_aug:", image_aug.shape, image_aug.dtype)
        imgaug_copy = image_aug.copy()
        # assert(len(Alphas) == (len(kpsoi_aug.keypoints)))
        for i, _ in enumerate(kpsoi_aug.keypoints):
            if i % 2 == 1:
                continue
            kp = kpsoi_aug.keypoints[i]
            x0, y0 = kp.x, kp.y
            kp = kpsoi_aug.keypoints[i+1]
            x1, y1 = kp.x, kp.y

            alpha = get_alpha(x0, y0, x1, y1)
            # beta = round(math.degrees(-math.acos(-math.cos(math.radians(Alphas[i]))))) + 360
            # alpha = beta % 180 if Alphas[i] > 180 else beta
            new_annots[i//2,
                       1:NUM_VARIABLES+1] = x0, y0, alpha # abs(180 - alpha)
        print("image_aug", image_aug.shape, image_aug.dtype)
        x_in_bound = np.logical_and(
            new_annots[:, 1] >= 0, new_annots[:, 1] < image_aug.shape[1])
        y_in_bound = np.logical_and(
            new_annots[:, 2] >= 0, new_annots[:, 2] < image_aug.shape[0])
        in_bound = np.logical_and(x_in_bound, y_in_bound)

        new_annots = new_annots[in_bound, :]
        # for x, y, alpha in new_annots:
        #     imgaug_copy = std_draw_line(
        #         imgaug_copy,
        #         point=(x, y),
        #         alpha=alpha,
        #         mode=DrawMode.Accept
        #     )
        # cv.imwrite('./aug_imgs/{}.png'.format(
        #     np.random.randint(0, 1000)), imgaug_copy)
        smpl = {'img': image_aug, 'annot': new_annots}

        return smpl


def detect_one_image(retinanet_model, data) -> list:
    scale = data['scale']
    img_name = float(int(data["name"]))

    # run network
    if torch.cuda.is_available():
        scores, labels, boxes = retinanet_model(data['img'].permute(
            2, 0, 1).cuda().float().unsqueeze(dim=0))
    else:
        scores, labels, boxes = retinanet_model(
            data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    boxes = boxes.cpu().numpy()
    if boxes.shape[0] == 0:
        return []
    # correct boxes for image scale
    boxes /= scale

    # select detections
    image_boxes = boxes
    image_scores = scores
    image_labels = labels
    img_name_col = np.full(shape=(len(image_scores), 1), fill_value=img_name, dtype=np.int32)
    image_detections = np.concatenate([img_name_col, image_boxes, np.expand_dims(
        image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
    return image_detections.tolist()

def detect(dataset, retinanet_model) -> dict:
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = list()
    all_augmented_detections = list()

    retinanet_model.eval()

    print("detecting")
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            print(data["img"].shape)
            image_detections = detect_one_image(retinanet_model=retinanet_model, data=data)
            all_detections.extend(image_detections)
            augmented_image_detections = augment_detector(retinanet_model=retinanet_model, data=data)
            all_augmented_detections.extend(augmented_image_detections)
            print('\rimage {0:02d}/{1:02d}'.format(index + 1, len(dataset)), end='')
    print()
    det = {
        'default': np.asarray(all_detections, dtype=np.float64),
        'augmented': np.asarray(all_augmented_detections, dtype=np.float64),
    }
    return det


def augment_detector(data, retinanet_model):
    global aug_detection_number
    aug_detection_number += 1
    aug = Augmenter()
    augmented_data = {'img': data['aug_img'], 'name': data['name'], 'scale': data['scale']}
    det = detect_one_image(retinanet_model=retinanet_model, data=augmented_data)
    if len(det) > 0:
        sample = {'img': data["only_aug_img"], 'annot': np.array(det)}
        img = data["only_aug_img"][:, :, ::-1]
        annot = sample['annot']
        img = draw(img=img, boxes=annot, augmented=True)
        save_path = osp.join(arguments.save_dir, "aug" + str(aug_detection_number) + arguments.ext)
        cv.imwrite(save_path, img)
        sample = aug(sample=sample)
        det = sample['annot'].tolist()
    return det


def load_random_sample(annotations, imgs_dir, ext=".jpg"):
    imgname = random.choice(list(annotations.keys()))
    imgpath = osp.join(imgs_dir, imgname + ext)
    img = cv.imread(imgpath)
    annotation = annotations[imgname]
    sample = {'img': img, 'annot': annotation}
    return sample


def draw(img, boxes, augmented=False):
    im = img.copy()
    for box in boxes:
        x, y, alpha = box[1:NUM_VARIABLES+1]
        p = (x, y)
        center_color = (0, 0, 0)
        if augmented:
            line_clor = (0, 0, 255)
        else:
            line_clor = (255, 0, 0)
        im = draw_line(image=im, p=p, alpha=alpha, line_color=line_clor, center_color=center_color, half_line=True)
    return im


def main(args):
    if osp.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    loader = imageloader.CSVDataset(
        filenames_path="annotations/filenames.json",
        partition="supervised",
        class_list="annotations/labels.csv",
        images_dir=args.image_dir,
        image_extension=args.ext,
        transform=torchvision.transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
    )
    fileModelIO = open(args.model, "rb")
    loaded_model = torch.load(fileModelIO)
    detections = detect(dataset=loader, retinanet_model=loaded_model)

    print('writting images:')
    for index in range(len(loader)):
        data = loader[index]
        img_name = data['name']
        img_load_path = osp.join(args.image_dir, img_name + args.ext)
        img_save_path = osp.join(args.save_dir, img_name + args.ext)
        img = cv.imread(img_load_path)
        img_name = int(img_name)
        default_det = detections['default'][detections['default'][:, 0] == img_name]
        aug_det = detections['augmented'][detections['augmented'][:, 0] == img_name]
        img = draw(img=img, boxes=default_det, augmented=False)
        img = draw(img=img, boxes=aug_det, augmented=True)
        cv.imwrite(img_save_path, img)
        print('\rdone {0}'.format(index), end='')
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-dir", type=str, required=True, dest="image_dir",
                        help="The directory where images are in.")
    parser.add_argument("-e", "--extension", type=str, required=False, dest="ext", default=".jpg",
                        choices=[".jpg", ".png"], help="image extension")
    parser.add_argument("-m", "--model", required=True, type=str, dest="model",
                        help="path to the model")
    parser.add_argument("-o", "--save-dir", type=str, required=True, dest="save_dir",
                        help="where to save output")
    arguments = parser.parse_args()
    main(args=arguments)
