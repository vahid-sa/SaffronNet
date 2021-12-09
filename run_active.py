import os
import shutil
import logging
from os import path as osp
from utils.training_tools import ActiveTraining
import retinanet


logging.basicConfig(level=logging.DEBUG)
retinanet.settings.NUM_QUERIES = 100
retinanet.settings.NOISY_THRESH = 0.5
# retinanet.settings.DAMPENING_PARAMETER = 0.0


class parser:
    ground_truth_annotations = osp.abspath("annotations/unsupervised.csv")
    csv_classes = osp.abspath("annotations/labels.csv")
    images_dir = osp.expanduser("~/Saffron/dataset/Train/")
    corrected_annotations = osp.expanduser("~/Saffron/active_annotations/corrected.csv")
    filenames = osp.abspath("annotations/filenames.json")
    partition = "unsupervised"
    ext = ".jpg"
    model_path = osp.expanduser('~/Saffron/weights/supervised/init_model.pt')
    state_dict_path = osp.expanduser('~/Saffron/init_fully_trained_weights/init_state_dict.pt')
    states_dir = osp.expanduser("~/Saffron/active_annotations/states")
    active_annotations = osp.expanduser("~/Saffron/active_annotations/train.csv")
    save_directory = osp.expanduser("~/tmp/saffron_imgs/")
    epochs = 30
    csv_val = osp.abspath("./annotations/validation.csv")
    save_models_directory = osp.expanduser("~/Saffron/weights/active")
    cycles = 10
    budget = 100
    supervised_annotations = osp.abspath("./annotations/supervised.csv")
    metrics_path = osp.expanduser("~/Safffron/metrics.json")

    @staticmethod
    def reset():
        if osp.isfile(parser.corrected_annotations):
            os.remove(parser.corrected_annotations)
        if osp.isfile(parser.active_annotations):
            os.remove(parser.active_annotations)
        if osp.isdir(parser.states_dir):
            shutil.rmtree(parser.states_dir)
        os.makedirs(parser.states_dir, exist_ok=False)
        if osp.isdir(parser.save_directory):
            shutil.rmtree(parser.save_directory)
        os.makedirs(parser.save_directory, exist_ok=False)
        if osp.isdir(parser.save_models_directory):
            shutil.rmtree(parser.save_models_directory)
        os.makedirs(parser.save_models_directory, exist_ok=False)
        if osp.isfile(parser.metrics_path):
            os.remove(parser.metrics_path)


def main():
    parser.reset()
    active_trainer = ActiveTraining(
        active_annotations_path=parser.active_annotations,
        corrected_annotations_path=parser.corrected_annotations,
        validation_file_path=parser.csv_val,
        groundtruth_annotations_path=parser.ground_truth_annotations,
        classes_path=parser.csv_classes,
        states_dir=parser.states_dir,
        images_dir=parser.images_dir,
        metrics_path=parser.metrics_path,
        supervised_annotations_path=parser.supervised_annotations,
        filenames_path=parser.filenames,
        epochs=parser.epochs,
    )
    active_trainer.run_cycle(
        cycles=parser.cycles,
        init_state_dict_path=parser.state_dict_path,
        models_directory=parser.save_models_directory,
        results_dir=parser.save_directory,
        budget=parser.budget,
    )


if __name__ == "__main__":
    main()
