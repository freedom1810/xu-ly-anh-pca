import cv2
import numpy as np 
import os


def load_data(path  = 'dataset/', size = (128, 128)):


    out = []
    dir_ = []
    for folder in os.listdir(path):
        path_folder = path + folder
        for img_path in os.listdir(path_folder):

            dir_.append(path_folder + '/' + img_path)
            img = cv2.imread(path_folder + '/' + img_path)

            img = cv2.resize(img, size)
            img = img.flatten().astype(np.float64)
            out.append(img)

    return dir_, np.array(out)
    