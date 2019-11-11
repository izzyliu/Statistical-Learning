import os
import glob
from PIL import Image
from keras.preprocessing import image
import numpy as np
import time

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

load_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resized_dataset')
save_root = os.path.dirname(os.path.abspath(__file__))
categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
Y_label = 0
image_height = 64
image_width  = 64

for category in categories :
    load_path = os.path.join(load_root, category, '*g')
    files = glob.glob(load_path)

    dataset_X = []
    dataset_Y = []
    counter = 0

    for file in files :
        img = Image.open(file)
        temp = image.img_to_array(img)
        temp = np.reshape(temp, (-1,1)).tolist()
        dataset_X.append(temp)
        dataset_Y.append(Y_label)
        counter += 1

    print(categories[Y_label], "finished!")
    Y_label	+= 1

    dataset_X = np.array(dataset_X)
    dataset_X = np.reshape(dataset_X, (-1,image_height, image_width, 3))
    dataset_X = dataset_X / 255
    dataset_Y = np.array(dataset_Y)
    dataset_Y = np.reshape(dataset_Y, (1, -1))
    dataset_Y = convert_to_one_hot(dataset_Y, 6).T

    print ("dataset_X shape: " + str(dataset_X.shape))
    print ("dataset_Y shape: " + str(dataset_Y.shape))

    save_path = os.path.join(save_root, category + "_dataset_X.npy")
    np.save(save_path, dataset_X)
    save_path = os.path.join(save_root, category + "_dataset_Y.npy")
    np.save(save_path, dataset_Y)
