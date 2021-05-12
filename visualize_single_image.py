import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def draw_line(img, bbox):
    half_length = 10
    x, y, alpha = bbox
    rows, cols, ch = img.shape
    xx, yy = int(round(half_length * np.cos(alpha))), int(round(half_length * np.sin(alpha)))
    start_point = max(x - xx, 0), max(y - yy, 0)
    end_point = min(x + xx, cols - 1), min(y + yy, rows - 1)
    print(start_point, end_point)
    cv2.line(img=img, pt1=start_point, pt2=end_point, color=(0, 0, 255), thickness=3)



# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, model_path, class_list):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    for img_name in os.listdir(image_path):

        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)

        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))
        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()

            print(image.shape)
            scores, classification, transformed_anchors = model(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x, y, alpha = int(bbox[0]), int(bbox[1]), int(bbox[2])
                label_name = labels[int(classification[idxs[0][j]])]
                print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(image_orig, (x, y, alpha), caption)
                # cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                cv2.circle(image, (x, y), 5, color=(0, 0, 255), thickness=-1)
                draw_line(img=image, bbox=bbox)

            cv2.imshow('detections', image_orig)
            cv2.waitKey(0)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names (see README)')

    parser = parser.parse_args()

    detect_image(parser.image_dir, parser.model_path, parser.class_list)
