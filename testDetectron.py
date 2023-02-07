#!/usr/bin/env python3

from detectron2.engine import DefaultPredictor
import os, pickle
from utilsDetectron import *

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

predictor = DefaultPredictor(cfg)

if runVideo:
    img_path = "test/Cars20.png"
    on_image(img_path, predictor)
else:
    video_path = ""
    on_video(video_path, predictor)

