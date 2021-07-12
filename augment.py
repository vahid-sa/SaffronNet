import numpy as np
import torch
import torchvision
import cv2 as cv
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
            iaa.Flipud(1.0),  # horizontal flips
            # color jitter, only affects the image
            # iaa.AddToHueAndSaturation((-50, 50))
        ])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
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

        x_in_bound = np.logical_and(
            new_annots[:, 0] > 0, new_annots[:, 0] < image_aug.shape[1])
        y_in_bound = np.logical_and(
            new_annots[:, 1] > 0, new_annots[:, 1] < image_aug.shape[0])
        in_bound = np.logical_and(x_in_bound, y_in_bound)

        new_annots = new_annots[in_bound, :]
        for x, y, alpha in new_annots:
            imgaug_copy = std_draw_line(
                imgaug_copy,
                point=(x, y),
                alpha=alpha,
                mode=DrawMode.Accept
            )
        # cv.imwrite('./aug_imgs/{}.png'.format(
        #     np.random.randint(0, 1000)), imgaug_copy)
        smpl = {'img': image_aug, 'annot': new_annots}

        return smpl


def detect_one_image(retinanet_model, data):
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
        return None
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

def detect(dataset, retinanet_model):
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
    aug = Augmenter()
    augmented_data = copy.deepcopy(data)
    augmented_data['img'] = augmented_data['img'][:, ::-1, ::]
    det = detect_one_image(retinanet_model=retinanet_model, data=augmented_data)
    sample = aug(sample={'img': augmented_data['img'], 'annot': det})
    det = sample['annot']
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
        x, y, alpha = box[1:NUM_VARIABLES]
        p = (x, y)
        center_color = (0, 0, 0)
        if augmented:
            line_clor = (0, 0, 255)
        else:
            line_clor = (255, 0, 0)
        im = draw_line(image=im, p=p, alpha=alpha, line_color=line_clor, center_color=center_color, half_line=True)
    return im


def main(args):
    loader = imageloader.CSVDataset(
        filenames_path="annotations/filenames.json",
        partition="unsupervised",
        class_list="annotations/labels.csv",
        images_dir=args.image_dir,
        image_extension=args.ext,
        transform=torchvision.transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
    )
    fileModelIO = open(args.model_path, "rb")
    loaded_model = torch.load(fileModelIO)
    detections = detect(dataset=loader, retinanet_model=loaded_model)
    print("detections", detections.keys())


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-dir", type=str, required=True, dest="image_dir",
                        help="The directory where images are in.")
    parser.add_argument("-e", "--extension", type=str, required=False, dest="ext", default=".jpg",
                        choices=[".jpg", ".png"], help="image extension")
    parser.add_argument("-m", "--model", required=True, type=str, dest="model",
                        help="path to the model")
    arguments = parser.parse_args()
    main(args=arguments)






"""
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--image-dir", help="images directory", required=True, type=str, dest="image_dir")
args = parser.parse_args()
f = open("annotations/unsupervised.csv")
csv_reader = csv.reader(f, delimiter=',')
annots = dict()
for i, row in enumerate(csv_reader):
    print(i, end="    ")
    key = row[0]
    try:
        annot = np.asarray([[int(float(row[1])), int(float(row[2])), 90 - int(float(row[3]))], ], dtype=np.int64)
    except ValueError:
        print("\nrow\n",row)
        continue
    if not (key in annots.keys()):
        annots[key] = annot
    else:
        annots[key] = np.concatenate([annots[key], annot], axis=0)
f.close()

aug = Augmenter()
origin_sample = load_random_sample(annotations=annots, imgs_dir=args.image_dir)
aug_sample = aug(sample=origin_sample)
aug_aug_sample = aug(sample=aug_sample)

orig_img = draw(img=origin_sample['img'], boxes=origin_sample['annot'])
aug_img = draw(img=aug_sample['img'], boxes=aug_sample['annot'])
aug_aug_img = draw(img=aug_aug_sample['img'], boxes=aug_aug_sample['annot'])


cv.imshow("orig", orig_img)
cv.imshow("aug", aug_img)
cv.imshow("aug_aug", aug_aug_img)
cv.waitKey(0)
cv.destroyAllWindows()
"""
