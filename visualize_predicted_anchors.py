import argparse
import numpy as np
import torch
import torchvision
from retinanet import model
from prediction import imageloader
from visualize import draw_selected_ignored

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def convert_results_to_detections(
        scores: torch.Tensor,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        scale: float,
        img_name: float
) -> list:
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    boxes = boxes.cpu().numpy()
    if boxes.shape[0] == 0:
        return []
    # correct boxes for image scale
    boxes /= scale

    # select detections
    image_boxes = boxes
    image_scores = scores
    image_labels = labels
    img_name_col = np.full(shape=(len(image_scores), 1), fill_value=img_name, dtype=np.int32)
    image_detections = np.concatenate([img_name_col, image_boxes, np.expand_dims(
        image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
    return image_detections.tolist()


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
    main_all_detections = list()
    co_all_detections = list()

    retinanet_model.eval()

    print("detecting")
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            img_name = float(int(data["name"]))

            # run network
            d = data['img'].permute(2, 0, 1)
            d = d.to(device=device)
            d = d.float()
            d = d.unsqueeze(dim=0)
            results = retinanet_model(d)
            scores, labels, boxes = results['main']
            image_detections = convert_results_to_detections(
                scores=scores,
                labels=labels,
                boxes=boxes,
                scale=scale,
                img_name=img_name
            )
            main_all_detections.extend(image_detections)
            scores, labels, boxes = results['co']
            image_detections = convert_results_to_detections(
                scores=scores,
                labels=labels,
                boxes=boxes,
                scale=scale,
                img_name=img_name
            )
            co_all_detections.extend(image_detections)
            print('\rimage {0:02d}/{1:02d}'.format(index + 1, len(dataset)), end='')
    print()
    all_detections = {
        "main": np.array(main_all_detections, dtype=np.float64),
        "co": np.array(co_all_detections, dtype=np.float64),
    }
    return all_detections


def predict(loader: imageloader.CSVDataset):
    retinanet = model.vgg7(num_classes=loader.num_classes(), pretrained=True).to(device=device)
    retinanet = torch.nn.DataParallel(retinanet).to(device=device)
    optimizer = torch.optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    retinanet.training = False
    retinanet.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    scheduler.load_state_dict(state_dict['scheduler_state_dict'])
    det = detect(dataset=loader, retinanet_model=retinanet)
    return det


if __name__ == "__main__":
    parser = argparse.ArgumentParser("visualize predicted anchors")
    parser.add_argument("-i", "--image-dir", type=str, required=True, dest="image_dir",
                        help="The directory where images are in.")
    parser.add_argument("-e", "--extension", type=str, required=False, dest="ext", default=".jpg",
                        choices=[".jpg", ".png"], help="image extension")
    parser.add_argument("-s", "--state-dict", required=True, type=str, dest="state_dict",
                        help="path to the state_dict")
    parser.add_argument("--image-save-dir", type=str, required=True, dest="image_save_dir",
                        help="where to save images")
    arguments = parser.parse_args()
    img_loader = imageloader.CSVDataset(
        filenames_path="annotations/filenames.json",
        partition="supervised",
        class_list="annotations/labels.csv",
        images_dir=arguments.image_dir,
        image_extension=arguments.ext,
        transform=torchvision.transforms.Compose([imageloader.Normalizer(), imageloader.Resizer()]),
    )
    with open(arguments.state_dict, "rb") as fileIO:
        state_dict = torch.load(fileIO)
    detections = predict(loader=img_loader)
    draw_selected_ignored(
        loader=img_loader,
        detections=detections,
        images_dir=arguments.image_dir,
        output_dir=arguments.image_save_dir,
        ext=arguments.ext,
    )
