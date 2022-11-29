#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import math
import csv

class tridimensionalInfo():
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
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

    def getFaceLandmarks(self):
        self.faceCoordinates = []
        if self.results.face_landmarks:
            for id, lm in enumerate(self.results.face_landmarks):
                print(id, lm)


    def getPoseWorldLandmarks(self):
        self.poseCoordinates = []
        if self.results.pose_world_landmarks:
            for id, lm in enumerate(self.results.pose_world_landmarks.landmark):
                self.poseCoordinates.append(tridimensionalInfo(lm.x, lm.y, lm.z, lm.visibility))
        

    def visibilityCheck(self, number):
        if self.poseCoordinates != []:
            return self.poseCoordinates[number].visibility >= 0.98
        return False

    def printPointCoordinates(self, number):
        if self.visibilityCheck(number):
            print("ID :" + str(number))
            print("x: " + str(self.poseCoordinates[number].x))
            print("y: " + str(self.poseCoordinates[number].y))
            print("z: " + str(self.poseCoordinates[number].z))
        else:
            print("Visibility of point " + str(number) + " is too low")


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


def openCsvFile(headerTitle, filename):
    header = [headerTitle]
    f = open(filename, 'w')
    # create the csv writer
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    return f, writer


def closeCsvFile(f):
    f.close()


def printBodyMeasurements(writerShoulder, writerHip, writerTorso, writerRightArm, writerLeftArm):

    # Shoulder Length
    shoulderLength = detector.distanceBetweenPoints(11, 12)
    if shoulderLength != -1:
        writerShoulder.writerow([shoulderLength])
        print("Shoulder Length: " + str(shoulderLength))
    else:
        print("Shoulder Length is not available!")


    # Hip Length
    hipLength = detector.distanceBetweenPoints(23, 24)
    if hipLength != -1:
        writerHip.writerow([hipLength])
        print("Hip Length: " + str(hipLength))
    else:
        print("Hip Length is not available!")


    # Torso Length
    middlePoint_1 = detector.getMiddlePoint(11, 12)
    middlePoint_2 = detector.getMiddlePoint(23, 24)

    if middlePoint_1 != -1 and middlePoint_2 != -1:
        torsoLen = detector.distanceFormula(
            middlePoint_1[0],
            middlePoint_1[1],
            middlePoint_1[2],
            middlePoint_2[0],
            middlePoint_2[1],
            middlePoint_2[2],
        )
        writerTorso.writerow([torsoLen])
        print("Torso Length: "+ str(torsoLen))
    else:
        print("Torso Length is not available!")


    # Right Arm Length
    rightArmLength = detector.getArmLenght(12, 14, 16)
    if rightArmLength != -1:
        writerRightArm.writerow([rightArmLength])
        print("Right Arm Length: " + str(rightArmLength))
    else:
        print("Right Arm Length is not available!")

    # Left Arm Length
    leftArmLength = detector.getArmLenght(11, 13, 15)
    if leftArmLength != -1:
        writerLeftArm.writerow([leftArmLength])
        print("Left Arm Length: " + str(leftArmLength))
    else:
        print("Left Arm Length is not available!")




# Initialize Pose Detector
detector = holisticDetector()

# Open CSV File
distance = '4m'
f_shoulder, writerShoulder = openCsvFile("shoulder_length", 'shoulder_test_' + distance + '.csv')
f_hip, writerHip = openCsvFile("hip_length", 'hip_test_' + distance + '.csv')
f_torso, writerTorso = openCsvFile("torso_length", 'torso_test_' + distance + '.csv')
f_right_arm, writerRightArm = openCsvFile("right_arm_length", 'right_arm_test_' + distance + '.csv')
f_left_arm, writerLeftArm = openCsvFile("left_arm_length", 'left_arm_test_' + distance + '.csv')

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

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        img = detector.find(color_image, True, True, False, False)
        detector.getPoseWorldLandmarks()

        printBodyMeasurements(writerShoulder, writerHip, writerTorso, writerRightArm, writerLeftArm)

        # detector.getFaceLandmarks()

        cv2.imshow('RealSense', img)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
    # Close csv file
    closeCsvFile(f_shoulder)
    closeCsvFile(f_hip)
    closeCsvFile(f_torso)
    closeCsvFile(f_right_arm)
    closeCsvFile(f_left_arm)

