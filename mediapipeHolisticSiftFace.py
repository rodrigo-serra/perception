#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import math
import csv
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

class tridimensionalInfo():
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class bidimensionalInfo():
    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.visibility = visibility


class holisticDetector():
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpHolistic = mp.solutions.holistic
        self.holistic = self.mpHolistic.Holistic()

    def find(self, img, pose, face, rightHand, leftHand):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(imgRGB)
    
        if pose and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpHolistic.POSE_CONNECTIONS)

        if face and self.results.face_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.face_landmarks, self.mpHolistic.FACEMESH_TESSELATION)
            # self.mpDraw.draw_landmarks(img, self.results.face_landmarks, self.mpHolistic.FACEMESH_CONTOURS)

        if rightHand and self.results.right_hand_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)

        if leftHand and self.results.left_hand_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)

        return img


    def getFaceLandmarks(self, img):
        self.faceCoordinates = []
        h, w, c = img.shape
        if self.results.face_landmarks:
            for id, lm in enumerate(self.results.face_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.faceCoordinates.append((cx, cy))
            return True
        return False


    def getPoseWorldLandmarks(self):
        self.poseCoordinates = []
        if self.results.pose_world_landmarks:
            for id, lm in enumerate(self.results.pose_world_landmarks.landmark):
                self.poseCoordinates.append(tridimensionalInfo(lm.x, lm.y, lm.z, lm.visibility))


    def getPoseImgLandmarks(self, img):
        self.imgPoseCoordinates = []
        h, w, c = img.shape
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.imgPoseCoordinates.append(bidimensionalInfo(cx, cy, lm.visibility))
                # cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return img
        

    def visibilityCheck(self, number):
        if self.poseCoordinates != []:
            return self.poseCoordinates[number].visibility >= 0.98
        return False

    def imgVisibilityCheck(self, number):
        if self.imgPoseCoordinates != []:
            return self.imgPoseCoordinates[number].visibility >= 0.98
        return False

    def printPointCoordinates(self, number):
        if self.visibilityCheck(number):
            print("ID :" + str(number))
            print("x: " + str(self.poseCoordinates[number].x))
            print("y: " + str(self.poseCoordinates[number].y))
            print("z: " + str(self.poseCoordinates[number].z))
        else:
            print("Visibility of point " + str(number) + " is too low")

    def printImgPointCoordinates(self, number):
        if self.imgVisibilityCheck(number):
            print("ID :" + str(number))
            print("u: " + str(self.imgPoseCoordinates[number].x))
            print("v: " + str(self.imgPoseCoordinates[number].y))
        else:
            print("Visibility of point " + str(number) + " is too low")

    def returnImgPointCoordinates(self, number):
        if self.imgVisibilityCheck(number):
            return self.imgPoseCoordinates[number].x, self.imgPoseCoordinates[number].y
        else:
            print("Visibility of point " + str(number) + " is too low")
            return -1, -1


    def distanceBetweenPoints(self, num1, num2):
        if self.visibilityCheck(num1) and self.visibilityCheck(num2):
            return self.distanceFormula(self.poseCoordinates[num1].x, 
                                        self.poseCoordinates[num1].y, 
                                        self.poseCoordinates[num1].z, 
                                        self.poseCoordinates[num2].x, 
                                        self.poseCoordinates[num2].y, 
                                        self.poseCoordinates[num2].z)
        else:
            return -1

    def distanceFormula(self, x1, y1, z1, x2, y2, z2):
        dx = math.pow(x1 - x2, 2)
        dy = math.pow(y1- y2, 2)
        dz = math.pow(z1 - z2, 2)
        return math.sqrt(dx + dy + dz)

    def getMiddlePoint(self, num1, num2):
        if self.visibilityCheck(num1) and self.visibilityCheck(num2):
            x = (self.poseCoordinates[num1].x + self.poseCoordinates[num2].x) / 2
            y = (self.poseCoordinates[num1].y + self.poseCoordinates[num2].y) / 2
            z = (self.poseCoordinates[num1].z + self.poseCoordinates[num2].z) / 2
            return [x, y, z]
        else:
            return -1

    
    def getArmLenght(self, num1, num2, num3):
        arm_1 = self.distanceBetweenPoints(num1, num2)
        arm_2 = self.distanceBetweenPoints(num2, num3)

        if arm_1 != -1 and arm_2!= -1:
            return arm_1 + arm_2
        else:
            return -1








def applySift(color_image, applyCrop, ux, uy):
    # Square Radius
    r = 20
    if applyCrop:
        # Crop Image
        color_image = color_image[uy - r:uy + r, ux - r:ux + r, :]  
    # Convert image to grayscale for SIFT
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    # Extract SIFT features
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return color_image, gray_image, keypoints, descriptors

def drawKeypointsImg(gray_image, keypoints, color_image):
    # Draw Keypoints in image (single image)
    color_image = cv2.drawKeypoints(gray_image, keypoints, color_image)
    return color_image



def computeDistanceBetweenKeypoints(matches, prev_keypoints, keypoints):
    # Featured matched keypoints from images 1 and 2
    pts1 = np.float32([keypoints[m.queryIdx].pt for m in matches])
    pts2 = np.float32([prev_keypoints[m.trainIdx].pt for m in matches])

    # Convert x, y coordinates into complex numbers
    # so that the distances are much easier to compute
    z1 = np.array([[complex(c[0],c[1]) for c in pts1]])
    z2 = np.array([[complex(c[0],c[1]) for c in pts2]])

    # Computes the intradistances between keypoints for each image
    KP_dist1 = abs(z1.T - z1)
    KP_dist2 = abs(z2.T - z2)

    # Distance between featured matched keypoints
    FM_dist = abs(z2 - z1)
    print("Num of Matches: ")
    print(FM_dist.shape[1])
    print("Avg Distance Between Features (px): ")
    print(np.sum(FM_dist) / FM_dist.shape[1])


def applyMatching(descriptors, prev_descriptors, keypoints, prev_keypoints):
    # Match descriptors
    matches = bf.match(descriptors, prev_descriptors)
    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x:x.distance)

    # if len(matches) > 0:
    #     computeDistanceBetweenKeypoints(matches, prev_keypoints, keypoints)
    
    print("Num of Matches: ")
    print(len(matches))
    print("Avg Distance Between Features (not in px): ")
    print(sum(m.distance for m in matches)/len(matches))

    return matches


def siftKeypointsMatching(ux, uy, color_image, prev_descriptors, prev_keypoints, i, drawKeypoints):
    color_image, gray_image, keypoints, descriptors = applySift(color_image, False, ux, uy)
    # Find Matches Between Current Frame and Previous Frame
    # Showed Them Live
    if i != 0:
        matches = applyMatching(descriptors, prev_descriptors, keypoints, prev_keypoints)
        
        # Draw Keypoints in image (previous and current frame keypoints)
        if drawKeypoints:
            color_image = cv2.drawMatches(gray_image, 
                                        keypoints, 
                                        gray_image, 
                                        prev_keypoints, 
                                        matches[:len(matches)], 
                                        None, 
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    prev_descriptors = descriptors
    prev_keypoints = keypoints
    i += 1
    return color_image, prev_descriptors, prev_keypoints, i






# Initialize Pose Detector
detector = holisticDetector()

# Initialize SIFT
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



firstTimeRunning = 0
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
            height, width, c = img.shape
            mask_img = Image.new('L', (width, height), 0)

            # Draw mask based on polygon
            points = detector.faceCoordinates
            points = np.array(points)

            hull = ConvexHull(points)
            polygon = []
            for v in hull.vertices:
                polygon.append((points[v, 0], points[v, 1]))

            ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
            mask = np.array(mask_img)

            # Focus only on the landmarks for the mask
            # mask = np.array(mask_img)
            # for facepoint in detector.faceCoordinates:
            #     u, v = facepoint
            #     mask[v][u] = 255
            
            color_image = cv2.bitwise_and(img, img, mask=mask)

        
            # Test SIFT and detect features in the current frame
            # color_image, gray_image, keypoints, descriptors = applySift(color_image, False, -1, -1)
            # color_image = drawKeypointsImg(gray_image, keypoints, color_image)
            # img = color_image
        
            # Test Feature Matching For Shoulder Between the Current and the Previous Frame
            if firstTimeRunning == 0:
                img, prev_descriptors, prev_keypoints, i = siftKeypointsMatching(-1, -1, color_image, [], [], 0, False)
                firstTimeRunning += 1
            else:
                img, prev_descriptors, prev_keypoints, i = siftKeypointsMatching(-1, -1, color_image, prev_descriptors, prev_keypoints, i, False)



        cv2.imshow('RealSense', img)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
    # Close csv file
    # closeCsvFile(f_keypoints)
    # closeCsvFile(f_descriptors)

