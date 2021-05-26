import unittest
import torchvision
import numpy as np
import torch
from retinanet.settings import NUM_VARIABLES
from predict import imageloader


class TestLoader(unittest.TestCase):
    """ Test Anchor's functions functionality
    """
    def test_imageloader(self):
        filenames_path = "../annotations/filenames.json"
        partition = "unsupervised"
        class_list = "../annotations/labels.csv"
        images_dir = "/mnt/2tra/saeedi/Saffron/Train"
        image_extension = ".jpg"
        transform = torchvision.transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()])
        loader = imageloader.CSVDataset(
            filenames_path=filenames_path,
            partition=partition,
            class_list=class_list,
            images_dir=images_dir,
            image_extension=image_extension,
            transform=transform,
        )
        model_path = "../model_final.pt"
        retinanet = torch.load(model_path)

        use_gpu = True

        if use_gpu:
            if torch.cuda.is_available():
                retinanet = retinanet.cuda()

        # if torch.cuda.is_available():
        #     retinanet.load_state_dict(torch.load(model_path))
        #     retinanet = torch.nn.DataParallel(retinanet).cuda()
        # else:
        #     retinanet.load_state_dict(torch.load(model_path))
        #     retinanet = torch.nn.DataParallel(retinanet)

        retinanet.training = False
        retinanet.eval()
        retinanet.module.freeze_bn()

        all_detections = self._get_detections(loader, retinanet)
        print(all_detections)

    @staticmethod
    def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
        """ Get the detections from the retinanet using the generator.
        The result is a list of lists such that the size is:
            all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
        # Arguments
            dataset         : The generator used to run images through the retinanet.
            retinanet           : The retinanet to run on the images.
            score_threshold : The score confidence threshold to use.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save the images with visualized detections to.
        # Returns
            A list of lists containing the detections for each image in the generator.
        """
        all_detections = [[None for i in range(
            dataset.num_classes())] for j in range(len(dataset))]

        retinanet.eval()

        with torch.no_grad():

            for index in range(len(dataset)):
                data = dataset[index]
                scale = data['scale']

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

                # correct boxes for image scale
                boxes /= scale

                # select indices which have a score above the threshold
                indices = np.where(scores > score_threshold)[0]
                if indices.shape[0] > 0:
                    # select those scores
                    scores = scores[indices]

                    # find the order with which to sort the scores
                    scores_sort = np.argsort(-scores)[:max_detections]

                    # select detections
                    image_boxes = boxes[indices[scores_sort], :]
                    image_scores = scores[scores_sort]
                    image_labels = labels[indices[scores_sort]]
                    image_detections = np.concatenate([image_boxes, np.expand_dims(
                        image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                    # copy detections to all_detections
                    for label in range(dataset.num_classes()):
                        all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
                else:
                    # copy detections to all_detections
                    for label in range(dataset.num_classes()):
                        all_detections[index][label] = np.zeros(
                            (0, NUM_VARIABLES + 1))

                print('{}/{}'.format(index + 1, len(dataset)), end='\r')

        return all_detections

