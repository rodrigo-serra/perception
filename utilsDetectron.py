#!/usr/bin/env python3

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode

import random, cv2, json
import matplotlib.pyplot as plt


def plot_samples(dataset_name, n = 1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata = dataset_custom_metadata, scale = 0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize = (15, 20))
        plt.imshow(v.get_image())
        plt.show()


def save_dataset_metadata(dataset_name, save_metadata_dir):
    dataset_metadata = MetadataCatalog.get(dataset_name)
    dictionary = {
        "evaluator_type": dataset_metadata.evaluator_type,
        "image_root": dataset_metadata.image_root,
        "json_file": dataset_metadata.json_file,
        "name": dataset_metadata.name,
        "thing_classes": dataset_metadata.thing_classes,
        "thing_dataset_id_to_contiguous_id": dataset_metadata.thing_dataset_id_to_contiguous_id
    }
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
    
    # Writing to sample.json
    with open(save_metadata_dir, "w") as outfile:
        outfile.write(json_object)



def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()
    
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file_path)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32   # The "RoIHead batch size" (default: 512)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg


def on_image(image_path, predictor, dataset_metadata):
    img = cv2.imread(image_path)
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], metadata = dataset_metadata, scale = 0.5, instance_mode = ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (14, 10))
    plt.imshow(v.get_image())
    plt.show()


def on_video(video_path, predictor):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() == False:
        print("Error opening video...")
        return

    sucess, img = cap.read()
    while sucess:
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata = {}, scale = 0.5, instance_mode = ColorMode.SEGMENTATION)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Result", output.get_image()[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
