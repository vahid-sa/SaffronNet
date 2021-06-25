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


labels_path = "../annotations/labels.csv"
class_to_index, index_to_class = load_classes(csv_class_list_path=labels_path)


def get_annotations(annot_path):
    fileIO = open(annot_path, "r")
    csv_reader = csv.reader(fileIO, delimiter=',')

    annots = list()
    for row in csv_reader:
        try:
            img_idx = int(row[0])
            x = float(row[1])
            y = float(row[2])
            angle = float(row[3])
            label = class_to_index[row[4]]
            gt_status = get_gt_status(annot_row=row)
            annotation = img_idx, x, y, angle, label, gt_status
            annots.append(annotation)
        except ValueError:
            continue
    annots = np.asarray(annots, dtype=np.float64)
    fileIO.close()
    return annots


def get_gt_status(annot_row):
    if len(annot_row) == 6:
        annot_status = annot_row[5]
        if annot_status == "corrected":
            gt_status = 1
        elif annot_status == "noisy":
            gt_status = 0
        else:
            raise AssertionError("not 'noisy' or 'corrected'")
    elif len(annot_row) == 5:
        gt_status = 1
    else:
        raise AssertionError("Annotation format is incorrect")

    return gt_status


def generate_random_backgrounds(images_indices):
    x = np.random.randint(1296, size=(len(images_indices), 1))
    y = np.random.randint(972, size=(len(images_indices), 1))
    angle = np.random.randint(360, size=(len(images_indices), 1))
    label = np.full(shape=(len(images_indices), 1), dtype=np.int64, fill_value=-1)
    gt_status = np.ones(shape=(len(images_indices), 1), dtype=np.int64)
    img_idx = np.expand_dims(images_indices, axis=1).astype(np.int64)
    bg_annots = np.concatenate([img_idx, x, y, angle, label, gt_status], axis=1).astype(np.float64)
    return bg_annots


def write_annotations(annots, path):
    fileIO = open(path, "w")
    csv_writer = csv.writer(fileIO, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for annot in annots:
        img_idx = "{0:03d}".format(int(annot[0]))
        x = str(annot[1])
        y = str(annot[2])
        angle = str(annot[3])
        label = index_to_class[str(int(annot[4]))]
        status = "corrected" if annot[5] == 1 else "noisy"
        row = [img_idx, x, y, angle, label, status]
        csv_writer.writerow(row)

print("index_to_class", index_to_class)
annotation_path = "../annotations/unsupervised.csv"
annotations = get_annotations(annot_path=annotation_path)

img_indices = np.unique(annotations[:, 0].astype(np.int64))
bg_annots = generate_random_backgrounds(images_indices=img_indices)
annotations = np.concatenate([annotations, bg_annots], axis=0)

annotations = annotations[list(annotations[:, 0].astype(np.int64).argsort())]
write_annotations(annots=annotations, path="../annotations/test_background.csv")
