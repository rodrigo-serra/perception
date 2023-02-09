#!/usr/bin/env python3

from detectron2.engine import DefaultPredictor
import os, pickle, json
from utilsDetectron import *
# from detectron2.data.datasets import register_coco_instances

instanceSegmentation = True
runVideo = False

if instanceSegmentation == True:
    # Instance Segmentation (IS)
    cfg_save_path = "IS_cfg.pickle"
else:
    # Object Detection (OD)
    cfg_save_path = "OD_cfg.pickle"


with open(cfg_save_path, "rb") as f:
    cfg = pickle.load(f)


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 

############################
# OPTION 1) LOAD DATASET
# train_dataset_name = "LP_train"
# train_images_path = "train"
# train_json_annot_path = "train.json"
# register_coco_instances(name = train_dataset_name, metadata = {}, json_file = train_json_annot_path, image_root = train_images_path)

# OPTION 2) SET THE THINGS LIKE CLASSES MANUALLY
# MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = ["pringles"]

# OPTION 3) LOAD JSON FILE WITH THE METADATA
metadata_save_path = "dataset_metadata.json"
with open(metadata_save_path, 'r') as openfile:
    jsonObj = json.load(openfile)

MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).evaluator_type = jsonObj["evaluator_type"]
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).image_root = jsonObj["image_root"]
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).json_file = jsonObj["json_file"]
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = jsonObj["thing_classes"]
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id = jsonObj["thing_dataset_id_to_contiguous_id"]

# WHAT IT SHOULD PRINT
# Metadata(evaluator_type='coco', image_root='train', json_file='train.json', name='LP_train', thing_classes=['pringles'], thing_dataset_id_to_contiguous_id={0: 0})
# print(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
dataset_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
######################################

predictor = DefaultPredictor(cfg)

if not runVideo:
    img_path = "test/img_25.png"
    on_image(img_path, predictor, dataset_metadata)
else:
    video_path = ""
    on_video(video_path, predictor)

