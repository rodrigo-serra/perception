#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageEnhance

def getColorName(R,G,B):
    # Read CSV with color codes
    file_name = 'basic_colors_simplified.csv'
    index=["color","color_name","hex","R","G","B"]
    csv = pd.read_csv(file_name, names=index, header=None)
  
    minimum = 10000
    cname = ""
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname


def readColor(img):
    # Find image center (in img x is width and y is height)
    apply_median = True
    h, w, c = img.shape
    cx = int(w / 2)
    cy = int(h / 2)
    offset = 20

    cx_left = cx - offset
    cx_right = cx + offset

    cy_top = cy - offset
    cy_bottom = cy + offset

    new_img = img[cx_left:cx_right, cy_top:cy_bottom, :]

    img_blue_channel = img[cx_left:cx_right, cy_top:cy_bottom, 0]
    img_green_channel = img[cx_left:cx_right, cy_top:cy_bottom, 1]
    img_red_channel = img[cx_left:cx_right, cy_top:cy_bottom, 2]
        
    if np.all(img_blue_channel != img_blue_channel):
        return False
    blue = np.average(img_blue_channel)
    blue = int(blue)

    if np.all(img_green_channel != img_green_channel):
        return False
    green = np.average(img_green_channel)
    green = int(green)
    
    if np.all(img_red_channel != img_red_channel):
        return False
    red = np.average(img_red_channel)
    red = int(red)
    
    # print("RGB CODE: " + str(red) + "; " + str(green) + "; " + str(blue))

    color_name = getColorName(red, green, blue)
    print("Color Name: " + color_name)

    return new_img


    
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

        applyContrast = False
        if applyContrast:
            # CONTRAST AND BRIGHTNESS
            alpha = 1.5 # Contrast control
            beta = 10 # Brightness control
            color_image = cv2.convertScaleAbs(color_image, alpha=alpha, beta=beta)

        applySaturation = True
        if applySaturation:
            # PIL, Creating object of Color class
            color_coverted = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_coverted)
            color_image = ImageEnhance.Color(pil_image)
            color_image = color_image.enhance(4.0)
            color_image = np.array(color_image)  
            color_image = color_image[:, :, ::-1].copy() # Convert RGB to BGR

        # color_image = readColor(color_image)
        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()




