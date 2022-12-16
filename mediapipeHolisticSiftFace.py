#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

from holisticDetectorModule import *
from siftModule import *
from csvModule import *

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

# Open CSV Files
f_matches, writerMatches = openCsvFile(["num_matches", "avg_distance"], 'matches.csv')

ux = -1
uy = -1
prev_descriptors = []
prev_keypoints = []
iterator = 0
drawKeypoints = False
cropImg = False
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
        img = detector.find(color_image, False, False, False, False)
        
        isFaceLandmarks = detector.getFaceLandmarks(img)
        if isFaceLandmarks:
            color_image = getFaceMask(img, detector.faceCoordinates)
        
            # Test SIFT and detect features in the current frame
            # color_image, gray_image, keypoints, descriptors = applySift(color_image, cropImg, ux, uy, radius)
            # color_image = drawKeypointsImg(gray_image, keypoints, color_image)
            # img = color_image
        
            # Test Feature Matching Between the Current and the Previous Frame
            img, prev_descriptors, prev_keypoints, iterator, matches = siftKeypointsMatching(ux, uy, color_image, prev_descriptors, prev_keypoints, iterator, drawKeypoints, cropImg, radius)
            
            if len(matches) > 0:
                num_matches = len(matches)
                avg_distance_matches = sum(m.distance for m in matches)/len(matches)
                print("Num of Matches: ")
                print(num_matches)
                print("Avg Distance Between Features (not in px): ")
                print(avg_distance_matches)
                # Write to CSV
                writerMatches.writerow([num_matches, avg_distance_matches])


        cv2.imshow('RealSense', img)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
    # Close csv file
    closeCsvFile(f_matches)
