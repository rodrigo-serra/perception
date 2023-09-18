#!/usr/bin/env python3

from deepface import DeepFace
import os, cv2
from utils.imgs_utils import ImgProcessingUtils
from utils.files_utils import FilesUtils

class ReadFace():
    def __init__(self):
        self.ipu = ImgProcessingUtils()
        self.fu = FilesUtils()
        self.models = ["VGG-Face", "Facenet", "Facenet512","OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
        self.backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]
        self.metrics = ["cosine", "euclidean", "euclidean_l2"]
        self.actions = ['age', 'gender', 'race', 'emotion']
        self.db_path = os.getcwd() + '/db/'

    
    def detectFace(self, img_path):
        # DeepFace Detect Face
        face_objs = DeepFace.extract_faces(
            img_path = img_path,
            target_size = (244, 244),
            enforce_detection = True,
            detector_backend = 'opencv',
            align = True
        )

        # Read img (OpenCV)
        img = self.ipu.readImg(img_path=img_path)
        raw_img = img.copy()

        if len(face_objs) != 0 and face_objs is not None:
            faces = []
            for face_obj in face_objs:
                # face_obj (img, 'facial_area': {'x': , 'y': , 'w': , 'h': }, 'confidence': })
                facial_area = face_obj["facial_area"]
                faces.append((
                    facial_area["x"],
                    facial_area["y"],
                    facial_area["w"],
                    facial_area["h"],
                ))

            detected_faces = []
            for x, y, w, h in faces:
                if w > 130:
                    detected_face = raw_img[int(y) : int(y + h), int(x) : int(x + w)]
                    self.ipu.drawRectangle(img, (x, y), (x + w, y + h), 'red', 3)
                    detected_faces.append((x, y, w, h))


            # For each face detected, it takes the cropped img and looks for the person in the dataset.
            # If person is in the db, saving the img is not required. Identify person by its name and show the confidence.
            # Otherwise, identify the person as unknow.
            # If requested, save photos of unknown persons to the db.



        # Display Img (OpenCV)
        # self.ipu.displayImg(img=img)
        self.ipu.displayImg(img=detected_face)
    

    def faceVerification(self, img1_path, img2_path, model_num=0):
        # DeepFace Verify Function
        res = DeepFace.verify(
            img1_path = img1_path,
            img2_path = img2_path,
            model_name = self.models[model_num],
            distance_metric = 'cosine',
            enforce_detection = True,
            detector_backend = 'opencv',
            align = True,
            normalization = 'base'
        )
        print(res)


    def stream(self):
        res = DeepFace.stream(
            db_path = self.db_path,
            model_name = self.models[0],
            distance_metric = 'cosine',
            detector_backend = 'opencv',
            enable_face_analysis = False,
            source = '/dev/video0',
            time_threshold = 2,
            frame_threshold = 2
        )
        print(res)

    
    def getImgFromDb(self, person, num, img_format):
        img_path = self.db_path + person + '/' + person + "_" + str(num) + '.' + img_format
        if self.fu.fileExists(img_path):
            return img_path
        
        print("The following img does not exist: " + img_path)
        return None



def main():
    df = ReadFace()

    ## Detect Face
    person = 'michael_phelps'
    img1 = df.getImgFromDb(person, 1, 'jpg')
    df.detectFace(img1)

    ## Face Verification
    # person = 'michael_phelps'
    # img1 = df.getImgFromDb(person, 1, 'jpg')
    # img2 = df.getImgFromDb(person, 2, 'jpg')
    
    # if img1 is None or img2 is None:
    #     exit(1)
    
    # df.faceVerification(img1, img2)

    ## Stream
    # df.stream()


if __name__ == "__main__":
    main()