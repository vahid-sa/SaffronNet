import numpy as np
import torch
import torchvision
import argparse
from prediction import imageloader, predict_boxes
import labeling


parser = argparse.ArgumentParser(description="Get required values for box prediction and labeling.")
parser.add_argument("-f", "--filename-path", required=True, type=str, dest="filenames_path",
                    help="Path to the file that reads the name of image files")
parser.add_argument("-p", "--partition", required=True, type=str, dest="partition",
                    choices=["supervised", "unsupervised", "validation", "test"], help="which part of file names")
parser.add_argument("-c", "--class-list", type=str, required=True, dest="class_list",
                    help="path to the class_list file")

parser.add_argument("-i", "--image-dir", type=str, required=True, dest="image_dir",
                    help="The directory where images are in.")
parser.add_argument("-e", "--extension", type=str, required=False, dest="ext", default=".jpg",
                    choices=[".jpg", ".png"], help="image extension")
parser.add_argument("-m", "--model", required=True, type=str, dest="model",
                    help="path to the model")
parser.add_argument("--annotations", type=str, required=True, dest="annotations",
                    help="path to the ground_truth annotations compatible with partition")
args = parser.parse_args()


def detect(dataset, retinanet):
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

    retinanet.eval()

    print("detecting")
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            img_name = float(int(data["name"]))

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = retinanet(data['img'].permute(
                    2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = retinanet(
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

loader = imageloader.CSVDataset(
    filenames_path=args.filenames_path,
    partition=args.partition,
    class_list=args.class_list,
    images_dir=args.image_dir,
    image_extension=args.ext,
    transform=torchvision.transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
)

model = torch.load(args.model)
pred_boxes = detect(dataset=loader, retinanet=model)
print("pred_boxes", pred_boxes.dtype, pred_boxes.shape)
