import torch
import numpy as np


def get_detections(dataset, retinanet):
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
