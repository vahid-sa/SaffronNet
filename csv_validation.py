import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--images_path',help='Path to images directory',type=str)
    parser.add_argument('--class_list_path',help='Path to classlist csv',type=str)
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
    parser = parser.parse_args(args)

    #dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(parser.csv_annotations_path,parser.class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]), images_dir=parser.images_path)
    state_dict = torch.load(parser.model_path)
    # Create the model
    retinanet = model.vgg7(num_classes=dataset_val.num_classes(), pretrained=True).to(device=device)
    retinanet = torch.nn.DataParallel(retinanet).to(device=device)
    retinanet.training = False
    # retinanet=torch.load(parser.model_path)
    retinanet.load_state_dict(state_dict['model_state_dict'])

    # use_gpu = True
    #
    # if use_gpu:
    #     if torch.cuda.is_available():
    #         retinanet = retinanet.cuda()

    # if torch.cuda.is_available():
    #     retinanet.load_state_dict(torch.load(parser.model_path))
    #     retinanet = torch.nn.DataParallel(retinanet).cuda()
    # else:
    #     retinanet.load_state_dict(torch.load(parser.model_path))
    #     retinanet = torch.nn.DataParallel(retinanet)
    #
    # retinanet.training = False
    # retinanet.eval()
    # retinanet.module.freeze_bn()

    print(csv_eval.evaluate(dataset_val, retinanet))



if __name__ == '__main__':
    main()
