#!/bin/bash
# example of using arguments to a script
echo "My first name is $1"
echo "My surname is $2"
echo "Total number of arguments is $#"


"""
cd /mnt/2tra/saeedi/Projects/SaffronNet/ || exit
# predict_boxes
/mnt/2tra/saeedi/venvs/retinanet/bin/python3.6 predict_boxes.py \
  --filenames_path annotations/filenames.json \
  --partition unsupervised \
  --image_dir /mnt/2tra/saeedi/Saffron/Train/ \
  --model_path model_final.pt \
  --class_list annotations/labels.csv \
  --ext .jpg
"""
