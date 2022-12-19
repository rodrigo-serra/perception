#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

from holisticDetectorModule import *
from siftModule import *
from csvModule import *
from facerecModule import *

import time

# Draw mask based on polygon
def getFaceMask(img, points):
    height, width, c = img.shape
    mask_img = Image.new('L', (width, height), 0)

    points = np.array(points)

    hull = ConvexHull(points)
    polygon = []
    for v in hull.vertices:
        polygon.append((points[v, 0], points[v, 1]))

    ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask_img)
    
    color_image = cv2.bitwise_and(img, img, mask=mask)

    return color_image



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

personCounter = 0

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

start_time = time.time()

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
        img = detector.find(color_image, False, False, False, False)
        
        isFaceLandmarks = detector.getFaceLandmarks(img)
        if isFaceLandmarks:
            img_masked = getFaceMask(img, detector.faceCoordinates)

            # First Detection
            if personCounter == 0 and exec_time > 2:
                print("ADDED FIRST PERSON!")
                imageRGB = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(imageRGB)
                directory = os.getcwd()
                im_name = directory + "/images/human_" + str(personCounter) + ".png"
                im.save(im_name)
                loaded_img = face_recognition.load_image_file(im_name)
                known_face_encodings.append(face_recognition.face_encodings(loaded_img)[0])
                known_face_names.append("Human #" + str(personCounter))
                personCounter += 1
            else:
                face_locations, face_names = faceRecognition(img, known_face_encodings, known_face_names)
                img = drawRectangleAroundFace(img, face_locations, face_names)


        exec_time = time.time() - start_time

        cv2.imshow('RealSense', img)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()