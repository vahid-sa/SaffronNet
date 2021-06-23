import numpy as np
import csv


def load_classes(csv_class_list_path: str):
    """
    loads class list defined in dataset
    :param csv_class_list_path: path to csv class list
    :return: a dict that converts class to index and a dict that convert index to class
    """
    index_to_class = dict()
    class_to_index = dict()
    fileIO = open(csv_class_list_path, "r")
    reader = csv.reader(fileIO, delimiter=",")
    for row in reader:
        class_name, str_class_index = row
        index_to_class[str_class_index] = class_name
        class_to_index[class_name] = str_class_index
    fileIO.close()
    return class_to_index, index_to_class

def get_gt_status(annot_row):
    if len(row) == 6:
        annot_status = row[5]
        if annot_status == "corrected":
            gt_status = 1
        elif annot_status == "noisy":
            gt_status = 0
        else:
            raise AssertionError("not 'noisy' or 'corrected'")
    elif len(row) == 5:
        gt_status = 1
    else:
        raise AssertionError("Annotation format is incorrect")

    return gt_status


labels_path = "../annotations/labels.csv"
class_to_index, index_to_class = load_classes(csv_class_list_path=labels_path)

annot_path = "../annotations/supervised.csv"
fileIO = open(annot_path, "r")
csv_reader = csv.reader(fileIO, delimiter=',')

annotations = list()
for row in csv_reader:
    img_idx = int(row[0])
    x = float(row[1])
    y = float(row[2])
    angle = float(row[3])
    label = class_to_index[row[4]]
    gt_status = get_gt_status(annot_row=row)
    annotation = img_idx, x, y, angle, label, gt_status
    annotations.append(annotation)
annotations = np.asarray(annotations, dtype=np.float64)
fileIO.close()

img_indices = np.unique(annotations[:, 0].astype(np.int64))
print(class_to_index, index_to_class)