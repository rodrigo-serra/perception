#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import os

from facerecModule import *



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
# Load a sample picture and learn how to recognize it.

directory = os.getcwd()
dmytro_image = face_recognition.load_image_file(directory + "/images/dmytro.png")
dmytro_face_encoding = face_recognition.face_encodings(dmytro_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    dmytro_face_encoding
]
known_face_names = [
    "Dmytro"
]


try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert images to numpy arrays
        frame = np.asanyarray(color_frame.get_data())

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        face_locations, face_names = faceRecognition(frame, known_face_encodings, known_face_names)
        print(face_locations)
        print(face_names)
        frame = drawRectangleAroundFace(frame, face_locations, face_names)
                    
        cv2.imshow('RealSense', frame)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()