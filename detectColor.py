#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageEnhance
from colorthief import ColorThief
import os


def getColorName(R,G,B):
    # Read CSV with color codes
    file_name = 'basic_colors.csv'
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

def getColorNameHsv(h, s, b):
    if h < 5:
        if s < 51:
            if b < 51: return "gray"
            elif b < 102: return "gray"
            elif b < 153: return "gray"
            elif b < 204:return "gray"
            else: return "pink"
        elif s < 102:
            if b < 51: return "brown"
            elif b < 102: return "brown"
            elif b < 153: return "brown"
            elif b < 204: return "pink"
            else: return "pink"
        elif s < 153:
            if b < 51: return "brown"
            elif b < 102: return "brown"
            elif b < 153: return "brown"
            elif b < 204: return "brown"
            else: return "orange"
        elif s < 204:
            if b < 51: return "brown"
            elif b < 102: return "brown"
            elif b < 153: return "brown"
            elif b < 204: return "orange"
            else: return "orange"
        else:
            if b < 51: return "brown"
            elif b < 102: return "brown"
            elif b < 153: return "red"
            elif b < 204: return "red"
            else: return "red"
    elif h < 22:
        if s < 51:
            if b < 51: return "gray"
            elif b < 102: return "gray"
            elif b < 153: return "gray"
            elif b < 204: return "gray"
            else: return "white"
        elif s < 102:
            if b < 51: return "brown"
            elif b < 102: return "brown"
            elif b < 153: return "brown"
            elif b < 204: return "brown"
            else: return "yellow"
        elif s < 153:
            if b < 51: return "brown"
            elif b < 102: return "brown"
            elif b < 153: return "brown"
            elif b < 204: return "yellow"
            else: return "yellow"
        elif s < 204:
            if b < 51: return "brown"
            elif b < 102: return "brown"
            elif b < 153: return "brown"
            elif b < 204: return "yellow"
            else: return "yellow"
        else:
            if b < 51: return "brown"
            elif b < 102: return "brown"
            elif b < 153: return "brown"
            elif b < 204: return "orange"
            else: return "orange"
    elif h < 33:
        if s < 51:
            if b < 51: return "gray"
            elif b < 102: return "gray"
            elif b < 153: return "gray"
            elif b < 204: return "white"
            else: return "white"
        elif s < 102:
            if b < 51: return "gray"
            elif b < 102: return "green"
            elif b < 153: return "green"
            elif b < 204: return "green"
            else: return "white"
        elif s < 153:
            if b < 51: return "brown"
            elif b < 102: return "green"
            elif b < 153: return "green"
            elif b < 204: return "green"
            else: return "yellow"
        elif s < 204:
            if b < 51: return "brown"
            elif b < 102: return "green"
            elif b < 153: return "green"
            elif b < 204: return "green"
            else: return "yellow"
        else:
            if b < 51: return "brown"
            elif b < 102: return "green"
            elif b < 153: return "green"
            elif b < 204: return "yellow"
            else: return "yellow"
    elif h < 78:
        if s < 51:
            if b < 51: return "gray"
            elif b < 102: return "gray"
            elif b < 153: return "gray"
            elif b < 204: return "gray"
            else: return "white"
        elif s < 102:
            if b < 51: return "gray"
            elif b < 102: return "gray" #green
            elif b < 153: return "gray"
            elif b < 204: return "white"
            else: return "white"
        elif s < 153:
            if b < 51: return "gray"
            elif b < 102: return "green"
            elif b < 153: return "green"
            elif b < 204: return "white"
            else: return "white"
        elif s < 204:
            if b < 51: return "green"
            elif b < 102: return "green"
            elif b < 153: return "green"
            elif b < 204: return "green"
            else: return "white"
        else:
            if b < 51: return "green"
            elif b < 102: return "green"
            elif b < 153: return "green"
            elif b < 204: return "green"
            else: return "green"
    elif h < 131:
        if s < 51:
            if b < 51: return "gray"
            elif b < 102: return "gray" #purpleish
            elif b < 153: return "purple"
            elif b < 204: return "purple"
            else: return "white"
        elif s < 102:
            if b < 51: return "blue"
            elif b < 102: return "purple"
            elif b < 153: return "purple"
            elif b < 204: return "purple"
            else: return "purple"
        elif s < 153:
            if b < 51: return "blue"
            elif b < 102: return "purple"
            elif b < 153: return "purple"
            elif b < 204: return "purple"
            else: return "purple"
        elif s < 204:
            if b < 51: return "blue"
            elif b < 102: return "blue"
            elif b < 153: return "blue"
            elif b < 204: return "blue"
            else: return "purple"
        else:
            if b < 51: return "blue"
            elif b < 102: return "blue"
            elif b < 153: return "blue"
            elif b < 204: return "blue"
            else: return "blue"
    elif h < 167:
        if s < 51:
            if b < 51: return "gray"
            elif b < 102: return "gray"
            elif b < 153: return "gray"
            elif b < 204: return "pink"
            else: return "pink" #white
        elif s < 102:
            if b < 51: return "brown"
            elif b < 102: return "purple"
            elif b < 153: return "purple"
            elif b < 204: return "pink"
            else: return "pink"
        elif s < 153:
            if b < 51: return "brown"
            elif b < 102: return "purple"
            elif b < 153: return "purple"
            elif b < 204: return "pink"
            else: return "pink"
        elif s < 204:
            if b < 51: return "brown"
            elif b < 102: return "purple"
            elif b < 153: return "purple"
            elif b < 204: return "pink"
            else: return "pink"
        else:
            if b < 51: return "brown"
            elif b < 102: return "purple"
            elif b < 153: return "purple"
            elif b < 204: return "pink"
            else: return "pink"
    else:
        if s < 51:
            if b < 51: return "gray"
            elif b < 102: return "gray"
            elif b < 153: return "gray"
            elif b < 204: return "pink"
            else: return "pink" #white
        elif s < 102:
            if b < 51: return "brown"
            elif b < 102: return "brown"
            elif b < 153: return "brown"
            elif b < 204: return "pink"
            else: return "pink"
        elif s < 153:
            if b < 51: return "brown"
            elif b < 102: return "brown"
            elif b < 153: return "brown"
            elif b < 204: return "pink"
            else: return "pink"
        elif s < 204:
            if b < 51: return "brown"
            elif b < 102: return "brown"
            elif b < 153: return "red"
            elif b < 204: return "red"
            else: return "red"
        else:
            if b < 51: return "brown"
            elif b < 102: return "brown"
            elif b < 153: return "red"
            elif b < 204: return "red"
            else: return "red"


def readColorHsv(img):
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
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_hue = img_hsv[cx_left:cx_right, cy_top:cy_bottom, 0]
    img_saturation = img_hsv[cx_left:cx_right, cy_top:cy_bottom, 1]
    img_brightness = img_hsv[cx_left:cx_right, cy_top:cy_bottom, 2]

    if np.all(img_hue != img_hue):
        return False
    hue = np.average(img_hue)
    hue = int(hue)

    if np.all(img_saturation != img_saturation):
        return False
    saturation = np.average(img_saturation)
    saturation = int(saturation)
    
    if np.all(img_brightness != img_brightness):
        return False
    brightness = np.average(img_brightness)
    brightness = int(brightness)

    color_name = "Undefined"
    color_name = getColorNameHsv(hue, saturation, brightness)
    print(color_name)
    return new_img


def testHsv(img):
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

    # Playing with HSV
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixel_center = hsv_frame[cx, cy]
    hue_value, hue_saturation, hue_brightness = pixel_center[0], pixel_center[1], pixel_center[2]
    print("Hue Value: " + str(hue_value))
    print("Hue Saturation: " + str(hue_saturation))
    print("Hue Brightness: " + str(hue_brightness))
    cv2.circle(img, (cx, cy), 5, (255, 0, 0), 3)

    color_name = "Undefined"
    color_name = getColorNameHsv(hue_value, hue_saturation, hue_brightness)
    # if hue_value < 5:
    #     color_name = "RED"
    # elif hue_value < 22:
    #     color_name = "ORANGE"
    # elif hue_value < 33:
    #     color_name = "YELLOW"
    # elif hue_value < 78:
    #     color_name = "GREEN"
    # elif hue_value < 131:
    #     color_name = "BLUE"
    # elif hue_value < 167:
    #     color_name ="VIOLET"
    # else:
    #     color_name = "RED"

    print(color_name)
    return img



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
            color_image = color_image.enhance(2.0)
            color_image = np.array(color_image)  
            color_image = color_image[:, :, ::-1].copy() # Convert RGB to BGR

        # Find color img
        color_image = readColorHsv(color_image)
        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()






