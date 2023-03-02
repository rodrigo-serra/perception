#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
import os


def faceBox(faceNet, frame):
    h = frame.shape[0]
    w = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1, (277, 277), [104, 117, 123], swapRB = False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * w)
            y1 = int(detection[0, 0, i, 4] * h)
            x2 = int(detection[0, 0, i, 5] * w)
            y2 = int(detection[0, 0, i, 6] * h)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

    
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)


# Model Params
model_dir = os.getcwd() + "/models/"

faceProto = model_dir + "opencv_face_detector.pbtxt"
faceModel = model_dir + "opencv_face_detector_uint8.pb"

ageProto = model_dir + "age_deploy.prototxt"
ageModel = model_dir + "age_net.caffemodel"

genderProto = model_dir + "gender_deploy.prototxt"
genderModel = model_dir + "gender_net.caffemodel"


faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

padding = 20

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        
        # Detection
        frame, bboxs = faceBox(faceNet, color_image)

        for bbox in bboxs:
            if bbox[0] > 0 and bbox[1] > 0 and bbox[2] > 0 and bbox[3] > 0:
                # face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding,frame.shape[0] - 1), max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
                print(face.shape)
                blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            
                genderNet.setInput(blob)
                genderPred = genderNet.forward()
                gender = genderList[genderPred[0].argmax()]

                ageNet.setInput(blob)
                agePred = ageNet.forward()
                age = ageList[agePred[0].argmax()]

                label = "{},{}".format(gender, age)
                cv2.rectangle(frame,(bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1) 
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', frame)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()






