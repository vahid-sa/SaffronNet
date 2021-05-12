import numpy as np
import csv
from os import path as osp
import os
import glob
import csv

LABEL = "saffron"


def read_one_file_annotations(file_path):
    annots = list()
    fileIO = open(file_path, "r", newline="\n")
    reader = csv.reader(fileIO, delimiter=',')
    # file_id = int(osp.splitext(osp.basename(file_path))[0])
    for row in reader:
        annots.append(row)
    fileIO.close()
    annots = np.asarray(annots, dtype=np.float32).tolist()
    return annots


def read_all_file_annotations(file_paths):
    # all_annots = list()
    # for file_path in file_paths:
    #     file_annots = read_one_file_annotations(file_path)
    #     all_annots.append(file_annots)
    all_annots = dict()
    for file_path in file_paths:
        file_annots = read_one_file_annotations(file_path)
        file_id = int(osp.splitext(osp.basename(file_path))[0])
        all_annots[file_id] = file_annots
    return all_annots


def dict_to_list(boxes: dict, key):
    annotations = list()
    if len(boxes) > 0:
        for box in boxes:
            x, y, d = box
            annotations.append([format(key, "03d"), x, y, d, LABEL])
    else:
        x, y, d = '', '', ''
        class_name = ''
        annotations.append([format(key, "03d"), x, y, d, class_name])
    return annotations


def extend_annotations(keys, all_boxes):
    annotations = list()
    for key in keys:
        img_annots = dict_to_list(all_boxes[key], key)
        annotations.extend(img_annots)
    return annotations


def split_train_validation_test(supervised_train_coef, unsupervised_train_coef, validation_coef, test_coef, annots):
    assert (supervised_train_coef >= 0) and (unsupervised_train_coef >= 0) and (validation_coef >= 0) and (test_coef >= 0)
    assert (supervised_train_coef + unsupervised_train_coef + validation_coef + test_coef) == 1.0
    coefs = [0.0] * 3
    coefs[0] = supervised_train_coef
    coefs[1] = coefs[0] + unsupervised_train_coef
    coefs[2] = coefs[1] + validation_coef

    for i in range(len(coefs)):
        # coefs[i] = int(coefs[i] * len(annots))
        coefs[i] = int(coefs[i] * len(annots.keys()))
    # indices = np.arange(len(annots))
    indices = list(annots.keys())
    np.random.shuffle(indices)
    supervised_keys, unsupervised_keys, val_keys, test_keys = np.split(indices, coefs)
    supervised = extend_annotations(keys=sorted(supervised_keys), all_boxes=annots)
    unsupervised = extend_annotations(keys=sorted(unsupervised_keys), all_boxes=annots)
    val = extend_annotations(keys=sorted(val_keys), all_boxes=annots)
    test = extend_annotations(keys=sorted(test_keys), all_boxes=annots)
    return supervised, unsupervised, val, test


def save_csv(path, object):
    fileIO = open(path, "w")
    writer = csv.writer(fileIO)
    writer.writerows(object)
    fileIO.close()


root_dir = "/mnt/2tra/saeedi/Saffron/Train"
file_path_pattern = osp.join(root_dir, "*.csv")
file_paths = glob.glob(file_path_pattern)
annots = read_all_file_annotations(file_paths)

supervised, unsupervised, val, test = split_train_validation_test(
    supervised_train_coef=0.2,
    unsupervised_train_coef=0.6,
    validation_coef=0.1,
    test_coef=0.1,
    annots=annots)
if not osp.isdir("./annotations"):
    os.mkdir("./annotations")
save_csv(path="./annotations/supervised.csv", object=supervised)
save_csv(path="./annotations/unsupervised.csv", object=unsupervised)
save_csv(path="./annotations/validation.csv", object=val)
save_csv(path="./annotations/test.csv", object=test)
