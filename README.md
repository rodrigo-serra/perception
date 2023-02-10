# Scripts Description

## Scripts
facerecReidHolistic.py - Testing Reid with the face recognition algorithm. If a face is detected by the face recognition algorithm, I use the mediapipe holistic detector to detect the landmarks and to get a better image of the person (face boundary).

facerecReid.py - Testing Reid with the face recognition algorithm.

facecrec - Testing the face recognition algorithm.

mediapipeHolisticSiftFaceReidTest - The same as mediapipeHolisticSiftFaceReid however it uses the face recognition algorithm. The detection of the face landmarks impacts the face recognition process.

mediapipeHolisticSiftFaceReid - Tested the Reid using the metedology describe in mediapipeHolisticSiftFace. The Reid is based on a threshold.

mediapipeHolisticSiftFace - Testing mediapipe holistic detection with SIFT detection and matching. The code focus on the face as it uses the face landmarks to come up with a mask and extract the face boundaries.

mediapipeHolisticSift.py - Testing mediapipe holistic detection with SIFT detection and matching. The code focus on the Right Shoulder (id: 11).

mediapipeHolistic.py - Testing mediapipe holistic detection. Writing measurements to csv file.

mediapipePose.py - Testing mediapipe pose detection.

peopleTracker.py - Testing people tracking algorithm.

readRealsenseData.py - Realsense Video Streaming.

siftTest.py - Testing SIFT algorithm

readCsvMeasurements - Read csv files written in mediapipeHolistic.py.


## Modules
holisticDetectorModule.py - This script holds all the classes that are used for the holistic detection. It settles on the mediapipe library (https://google.github.io/mediapipe/solutions/holistic.html).

facerecModule.py - It has all the funtions used in face recognition. It is based on the face recognition library (https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py).

csvModule.py - Functions for opening and closing files.

siftModule.py - SIFT functions (keypoint detection and drawing, and matching).

