#!/usr/bin/env python3

import cv2
from utils.pandas_utils import PandasUtils

class ImgProcessingUtils():
    def __init__(self):
        self.gu = PandasUtils()
    
    def readImg(self, img_path):
        img = cv2.imread(img_path)
        return img
    
    def displayImg(self, img, window_name="Window"):
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def drawRectangle(self, img, start_point, end_point, color_name, thickness):
        color_tuple = self.gu.findColor(color_name=color_name)
        cv2.rectangle(img, start_point, end_point, color_tuple, thickness) 