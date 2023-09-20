#!/usr/bin/env python3

import os, json, sys
import numpy as np
import matplotlib.pyplot as plt


def fileExists(path, error_phrase):
    if not os.path.exists(path):
        print(error_phrase)
        exit(1)


def readJson(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)

    return data


def buildPlot(data_path):
    # Read Json Data (List of Dicionaries)
    legend = []
    for idx, data in enumerate(data_path):
        dic = readJson(data)
        x = np.array(dic["x_axis"]["data"])
        y = np.array(dic["y_axis"]["data"])

        y_avg = np.average(y)
        y_avg = np.round(y_avg, decimals = 3)
        str_avg = " (average: " + str(y_avg) + ")"
        
        if idx == 0:
            x_label = dic["x_axis"]["name"] + " (" + dic["x_axis"]["units"] + ")"
            y_label = dic["y_axis"]["name"] + " (" + dic["y_axis"]["units"] + ")"
        
        file_name = data.split("/")[-1]
        leg = file_name.split(".")[0] + str_avg
        legend.append(leg)

        plt.scatter(x, y)

    
    # Limits
    plt.xlim(0, 160)
    # plt.ylim(0.5, 2.0)
    plt.ylim(0, 1.8)
    
    # Title
    plt.title("LIDARS Current Measurement")
    
    # Labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Legend
    plt.legend(legend)

    #Adding text inside a rectangular box by using the keyword 'bbox'
    # plt.text(0, 1.80, 'Parabola $Y = x^2$', fontsize = 22, bbox = dict(facecolor = 'red', alpha = 0.5))
    
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("There are missing input arguments!")
        exit(1)
    
    number_of_plots = int(sys.argv[1])
    
    if len(sys.argv) < 2 + number_of_plots:
        print("There are missing input arguments!")
        exit(1)

    if len(sys.argv) > 2 + number_of_plots:
        print("There are too many input arguments!")
        exit(1)

    data_path = []

    for n in range(2, 2 + number_of_plots):
        fileExists(sys.argv[n], "The file " + sys.argv[n] + " does not exist!")
        data_path.append(sys.argv[n])


    buildPlot(data_path)    


if __name__ == "__main__":
    main()