#!/usr/bin/env python3

import numpy as np
import csv

test_sequence = ['shoulder', 'hip', 'torso', 'right_arm', 'left_arm']
filename = '_test_'
values = {
    1: [],
    2: [],
    3: [],
    4: []
}


for test in test_sequence:
    for i in range(1, 5):
        with open(test + filename + str(i) + 'm.csv', 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = np.array(list(reader)).astype(float)

        values[i].append(np.average(data))

        # print(headers)
        # print(test)
        # print("Distance to camera: " + str(i) + "m")
        # print("Number of Samples: " + str(data.size))
        # print("Average Length: " + str(np.average(data)))

        # print("-----------------------------------------")



# TABLE FORMAT
print ("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20}".format('Dist to Camera (m)','Shoulder','Hip','Torso', 'Right Arm', 'Left Arm'))
for k, v in values.items():
    shoulder_meas, hip_meas, torso_meas, right_arm_meas, left_arm_meas = v
    print ("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20}".format(k, shoulder_meas, hip_meas, torso_meas, right_arm_meas, left_arm_meas))
