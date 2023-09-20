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





# Future Ideas
### Hardware
* Install 3D LIDAR;
* Gimbal Arm;
* Kinova Arm;
* New compliant gripper that can wrap around objects (soft robotics);
* Bigger gripper to facilitate grabbing bigger objects (ex: cereal box);


### Software/Middleware
* Migrate to ROS2;


### Perception Pipeline
* **Dataset**
    * Buy YCB objects or ask Vislab to use theirs;
    * Add YCB objects to simulation + Train YCB objects in simulation. Eventually try to make the simulation photorealistic (Uni Bremen);
    * Do testbed dataset (RoboCup oriented);

* **REID**
    * 3D LIDAR reID;
    * Improve the current face recognition algorithm (DeepFace currently being tested);
    * Test Vicente's REID approach (pointcloud);


### Navigation
* 3D LIDAR SLAM + 3D Localization + 3D Navigation;
* People tracker (Trajectory People Follow MPC + local planner to avoid obstacles) (Vicente fez reid usando a pointcloud da pessoa);


### Manipulation
* Implement force feedback on the gripper;
* Pregrasp and Grasp pose dictionary for each object;


### Speech
* Add voice commands running by default for safety: stop to stop robot, home to send the robot to home motion, ...;
* Chatbot built using chatgpt so the robot can interact with people in a non scripted conversation (the robot would need to know facts about himself and the environment. This is the thesis of Veronica Spelbrink but she is using a grammar based chatbot);


### Semantic Map
* Improve semantic map by adjusting drawn surfaces when they are perceived;


### Interfaces
* Explainable robot actions: GUI + sound that clarifies what the robot is doing;













Explainable robot actions: GUI + sound that clarifies what the robot is doing;
People tracker (Trajectory People Follow MPC + local planner to avoid obstacles) (Vicente fez reid usando a pointcloud da pessoa);
Buy YCB objects + Add YCB objects to simulation + Train YCB objects in simulation;
Install 3D LIDAR;
3D LIDAR reID;
3D LIDAR SLAM + 3D Localization + 3D Navigation;
Gimbal Arm;
Implement force feedback on the gripper;
Improve semantic map by adjusting drawn surfaces when they are perceived;
Migrate to ROS2;
Kinova Arm;
Pregrasp and Grasp pose dictionary for each object;
New compliant gripper that can wrap around objects (soft robotics);
Bigger gripper to facilitate grabbing bigger objects (ex: cereal box);
Add voice commands running by default for safety: stop to stop robot, home to send the robot to home motion, ...;
Chatbot built using chatgpt so the robot can interact with people in a non scripted conversation (the robot would need to know facts about himself and the environment. This is the thesis of Veronica Spelbrink but she is using a grammar based chatbot);