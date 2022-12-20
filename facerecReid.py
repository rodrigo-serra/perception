#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import os

from PIL import Image, ImageDraw
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

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

personCounter = 0
process_this_frame = True

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

        if process_this_frame:
            face_locations, face_names = faceRecognition(frame, known_face_encodings, known_face_names)
            
            if face_locations != []:
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    if name == "Unknown":
                        # Draw Mask
                        mask = np.zeros(frame.shape[:2], dtype="uint8")
                        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
                        # cv2.imshow("Rectangular Mask", mask)
                        img_masked = cv2.bitwise_and(frame, frame, mask=mask)
                        # Save Img and Add to Enconder
                        imageRGB = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)
                        # Save new image of Person (not required)
                        im = Image.fromarray(imageRGB)
                        im_name = directory + "/images/human_" + str(personCounter) + ".png"
                        im.save(im_name)
                        # Load new Person to enconder
                        faceEnconder = face_recognition.face_encodings(imageRGB)
                        if len(faceEnconder) > 0:
                            known_face_encodings.append(faceEnconder[0])
                            known_face_names.append("Human #" + str(personCounter))
                            personCounter += 1
                    else:
                        print("Seeing " + name)

        
        process_this_frame = not process_this_frame
                                    
        boxes_frame = drawRectangleAroundFace(frame, face_locations, face_names)
        cv2.imshow('RealSense', boxes_frame)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()