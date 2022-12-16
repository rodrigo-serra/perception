#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2

from holisticDetectorModule import *
from siftModule import *
from csvModule import *

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


# VARIABLE INITIALIZATION
# Initialize Pose Detector
detector = holisticDetector()

# Open CSV Files
f_keypoints, writerKeypoints = openCsvFile(["keypoints"], 'keypointsBg1.csv')
f_descriptors, writerDescriptors = openCsvFile(["descriptors"], 'descriptorsBg1.csv')

ux = -1
uy = -1
prev_descriptors = []
prev_keypoints = []
iterator = 0
drawKeypoints = True
cropImg = True
radius = 20

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        img = detector.find(color_image, True, True, False, False)
        
        img = detector.getPoseImgLandmarks(img)
        # detector.printImgPointCoordinates(11)
        
        # Test Feature Matching For Shoulder Between the Current and the Previous Frame
        # Shoulder Info (Right Shoulder Img Prespective), ID :11
        ux, uy = detector.returnImgPointCoordinates(11)
        if ux != -1 and uy != -1:
            # Test SIFT and detect features in the current frame
            # color_image, gray_image, keypoints, descriptors = applySift(img, cropImg, ux, uy, radius)
            # img = drawKeypointsImg(gray_image, keypoints, color_image)

            img, prev_descriptors, prev_keypoints, iterator, matches = siftKeypointsMatching(ux, uy, color_image, prev_descriptors, prev_keypoints, iterator, drawKeypoints, cropImg, radius)

            # Write to CSV
            writerKeypoints.writerow([prev_keypoints])
            writerDescriptors.writerow([prev_descriptors])



        cv2.imshow('RealSense', img)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
    # Close csv file
    closeCsvFile(f_keypoints)
    closeCsvFile(f_descriptors)

