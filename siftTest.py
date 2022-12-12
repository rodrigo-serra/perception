#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2

# SIFT
sift = cv2.SIFT_create()

# Feature Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)

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

i = 0

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Crop Image
        color_image = color_image[200:250, 210:260, :]
        # print(np.shape(color_image))

        # Convert image to grayscale for SIFT
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Extract SIFT features
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        # print(np.shape(descriptors))

        # Draw Keypoints in image (single image)
        # color_image = cv2.drawKeypoints(gray_image, keypoints, color_image)

        # Find Matches Between Current Frame and Previous Frame
        # Showed Them Live
        if i != 0:
            # Match descriptors
            matches = bf.match(descriptors, prev_descriptors)
            # Sort them in the order of their distance
            matches = sorted(matches, key = lambda x:x.distance)
            print("Num of Matches: ")
            print(len(matches))
            print("Avg Distance Between Features: ")
            print(sum(m.distance for m in matches)/len(matches))
            # Draw Keypoints in image (previous and current frame keypoints)
            color_image = cv2.drawMatches(gray_image, 
                                        keypoints, 
                                        gray_image, 
                                        prev_keypoints, 
                                        matches[:600], 
                                        None, 
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        prev_descriptors = descriptors
        prev_keypoints = keypoints
        i += 1
        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()

