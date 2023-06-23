#!/usr/bin/env python3

from PIL import Image
import cv2
import os, json, sys
import numpy as np
from scipy.spatial import ConvexHull


def displayResult(img, polygon, show_mask=False, resize=True, desired_height=356, draw_polygon=True):
    if draw_polygon:
        pts = np.array(polygon)
        pts = pts.reshape((-1, 1, 2))
        isClosed = True
        color = (255, 0, 0)
        thickness = 2
        img = cv2.polylines(img, [pts], isClosed, color, thickness)

    if resize:
        img = resizeImg(img, desired_height=desired_height, mask=show_mask)

    showImage(img)


def resizeImg(img, desired_height, mask=False):
    if mask:
        height, width = img.shape
    else:
        height, width, _ = img.shape
    
    ratio = width / height
    new_height = desired_height
    new_width = int(ratio * new_height)
    img = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img


def showImage(img):
    cv2.imshow("Rectangular Mask", img)
    cv2.waitKey(0)

    while True:
        # it waits till we press a key
        key = cv2.waitKey(0)
        # if we press esc
        if key == 27:
            print('Closing Windows')
            cv2.destroyAllWindows()
            break
    
    cv2.waitKey(1)


def testMask(img, np_array, resize=True, desired_height=356):
    color_image = cv2.bitwise_and(img, img, mask=np_array)

    if resize:
        color_image = resizeImg(color_image, desired_height=desired_height)

    showImage(color_image)


def find_polygon(np_array, offset_columns=110):
    row_idx, columns_idx = np.where(np_array == 255)
    idxs = []
    for idx, r in enumerate(row_idx):
        idxs.append((columns_idx[idx] - offset_columns, r))
        
    idxs = np.array(idxs)

    hull = ConvexHull(idxs)
    polygon = []
    for v in hull.vertices:
        polygon.append((idxs[v, 0], idxs[v, 1]))

    return polygon


def write_json(json_template_path, new_json_path, img_arr, img_name, polygon, label_name):
    # Read Template
    with open(json_template_path) as json_file:
        data = json.load(json_file)

    # Convert Polygon
    new_polygon = []
    for point in polygon:
        new_polygon.append([int(point[0]), int(point[1])])

    # Update Content
    copy_data = data
    height, width = img_arr.shape
    copy_data["shapes"][0]["label"] = label_name
    copy_data["shapes"][0]["points"] = new_polygon
    copy_data["imagePath"] = img_name + ".jpg"
    copy_data["imageHeight"] = int(height)
    copy_data["imageWidth"] = int(width)
    copy_data["imageData"] = ""
    
    # Write Json
    json_object = json.dumps(copy_data, indent=4)
    with open(new_json_path, "w") as outfile:
        outfile.write(json_object)



def getLabel(im_path, pbm_file_path, json_template_path, new_json_path, name, label_name):
    # Read RGB image
    im = cv2.imread(im_path)
    height, width, _ = im.shape

    # Read the PBM file
    with open(pbm_file_path, "rb") as f:
        # Read the PBM file header (two lines)
        # print(f.readline())
        # print(f.readline())
        
        # Read the dimensions from the PBM file
    #     width, height = map(int, f.readline().split())

        # Calculate the number of bytes required to store the image data
        num_bytes = (width * height + 7) // 8

        # Read the image data as bytes
        img_bytes = f.read(num_bytes)

    # Convert the image data bytes to a NumPy array
    bit_array = np.unpackbits(np.frombuffer(img_bytes, dtype=np.uint8))
    bit_array = bit_array[:width * height]  # Truncate any extra bits

    # Replaciing 1 for 255 for the mask
    bit_array[bit_array == 1] = 255

    # Reshape the 1-dimensional bit array to a 2-dimensional NumPy array
    np_array = bit_array.reshape(height, width)

    # Cleaning the edges
    np_array[0:50][:] = 0
    # np_array[2808:height][:] = 0

    # Print the shape and content of the NumPy array
    # print("Shape:", np_array.shape)
    # print("Array content:\n", np_array)

    # Test mask on image
    # testMask(im, np_array)

    # Find Polygon
    polygon = find_polygon(np_array)

    # Display Image
    # displayResult(im, polygon, resize=True, desired_height=356, draw_polygon=True)
    # displayResult(np_array, polygon, show_mask=True, resize=True, desired_height=356, draw_polygon=True)

    # Write json
    write_json(json_template_path, new_json_path, np_array, name, polygon, label_name)



def fileExists(path, error_phrase):
    if not os.path.exists(path):
        print(error_phrase)
        exit(1)



def findFilesInDir(path_to_dir, extension):
    files = []
    for file in os.listdir(path_to_dir):
        if file.endswith(extension):
            files.append(file)

    return files


def main():
    if len(sys.argv) < 5:
        print("There are missing input arguments!")
        exit(1)

    path_to_rgb_imgs = sys.argv[1]
    path_to_masks_imgs = sys.argv[2]
    path_to_labelling_json_template = sys.argv[3]
    label_name = sys.argv[4]
    
    if len(sys.argv) == 6:
        num_images_to_label = int(sys.argv[5])
        partial_labelling = True
    else:
        num_images_to_label = -1
        partial_labelling = False

    fileExists(path_to_rgb_imgs, 'The path to the rgb images is not correct!')
    fileExists(path_to_masks_imgs, 'The path to the masks is not correct!')
    fileExists(path_to_labelling_json_template, 'The path to the labelling json template file is not correct!')

    rgb_imgs = findFilesInDir(path_to_rgb_imgs, '.jpg')
    masks = findFilesInDir(path_to_masks_imgs, '.pbm')

    for img in rgb_imgs:
        img_name = img.split(".")[0]

        if partial_labelling:
            img_num = int(img_name.split("_")[1])
            if img_num <= num_images_to_label:
                img_path = path_to_rgb_imgs + "/" + img
                pbm_path = path_to_masks_imgs + "/" + img_name + "_mask.pbm"
                label_json_path = path_to_rgb_imgs + "/" + img_name + ".json" 
                getLabel(img_path, pbm_path, path_to_labelling_json_template, label_json_path, img, label_name)
        else:
            img_path = path_to_rgb_imgs + "/" + img
            pbm_path = path_to_masks_imgs + "/" + img_name + "_mask.pbm"
            label_json_path = path_to_rgb_imgs + "/" + img_name + ".json" 
            getLabel(img_path, pbm_path, path_to_labelling_json_template, label_json_path, img, label_name)

if __name__ == "__main__":
    main()