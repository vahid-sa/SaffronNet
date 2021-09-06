import os
import numpy as np
import torch
import torchvision
from os import path as osp
import argparse
import retinanet
from prediction import imageloader


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

    retinanet_model.eval()

    print("detecting")
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
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
                continue
            # correct boxes for image scale
            boxes /= scale

            # select detections
            image_boxes = boxes
            image_scores = scores
            image_labels = labels
            img_name_col = np.full(shape=(len(image_scores), 1), fill_value=img_name, dtype=np.int32)
            image_detections = np.concatenate([img_name_col, image_boxes, np.expand_dims(
                image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
            all_detections.extend(image_detections.tolist())
            print('\rimage {0:02d}/{1:02d}'.format(index + 1, len(dataset)), end='')
    print()
    return np.asarray(all_detections, dtype=np.float64)


def main():
    imageloader.CSVDataset(
        filenames_path="annotations/filenames.json",
        partition="supervised",
        class_list="annotations/labels.csv",
        images_dir=args.image_dir,
        image_extension=args.ext,
        transform=torchvision.transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get required values for box prediction and labeling.")
    parser.add_argument("-i", "--image-dir", type=str, required=True, dest="image_dir",
                        help="The directory where images are in.")
    parser.add_argument("-e", "--extension", type=str, required=False, dest="ext", default=".jpg",
                        choices=[".jpg", ".png"], help="image extension")
    parser.add_argument("-m", "--model", required=True, type=str, dest="model",
                        help="path to the model")
    parser.add_argument("-s", "--state-dict", required=True, type=str, dest="state_dict",
                        help="path to the state_dict")
    # parser.add_argument("-o", "--save-dir", type=str, required=True, dest="save_dir",
    #                     help="where to save output")
    # parser.add_argument("-c", "--num-cycles", type=int, required=True, dest="num_cycles",
    #                     help="number of active cycles")
    # parser.add_argument("-d", "--depth", type=int, required=True, dest="depth",
    #                     choices=(18, 34, 50, 101, 52), default=50, help="ResNet depth")
    parser.add_argument("-p", "--epochs", type=int, required=True, dest="epochs",
                        default=20, help="Number of Epochs")
    # parser.add_argument("--image-save-dir", type=str, required=True, dest="image_save_dir",
    #                     help="where to save images")
    # parser.add_argument('--model-type', type=str, default="vgg", dest="model_type",
    #                     help='backbone for retinanet, must be "resnet" of "vgg"')
    # parser.add_argument('-f', '--dampening-factor', type=float, dest='dampening_param', required=True,
    #                     help='dampaening parameter')
    parser.add_argument('-q', "--num-queries", type=int, default=100, dest='num_queries',
                        help="number of Asking boxes per cycle")
    parser.add_argument('-n', '--noisy-thresh', type=float, required=True, dest='noisy_thresh',
                        help='noisy threshold')
    args = parser.parse_args()

    retinanet.settings.DAMPENING_PARAMETER = args.dampening_param
    retinanet.settings.NUM_QUERIES = args.num_queries
    retinanet.settings.NOISY_THRESH = args.noisy_thresh
