# Process dataset from jpg to csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Path to the dataset
root_path = './data'
image_path = './ClothingMini/train'
csv_path = './ClothingMini/train.csv'

image_size = 128
image_channels = 3

# joint path
image_path = os.path.join(root_path, image_path)
csv_path = os.path.join(root_path, csv_path)

labels_dict = {0:'dress', 1:'hat', 2:'longsleeve', 3:'outwear', 4:'pants', 5:'shirt', 6:'shoes', 7:'shorts', 8:'skirt', 9:'t-shirt'}
__dict__ = {v: k for k, v in labels_dict.items()}

dataset = pd.DataFrame(columns=['label'] + ['pixel' + str(i) for i in range(image_size * image_size * image_channels)])
# Get the list of all the images in the dataset directory
# Traverse through all the files, get the label as the directory name, and fit the index of the label in the labels_dict
for image in os.listdir(image_path):
    _path = os.path.join(image_path, image)
    label = image.split('_')[0]
    label_num = __dict__[label]
    for i in os.listdir(_path):
        # modify the image to 512*512, is the rario is not fix, padding with 0
        img = cv2.imread(os.path.join(_path, i))
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape(image_size, image_size, image_channels).astype('int32')
        # Flatten the image
        img = img.flatten()
        # Add the label and each pixel of the image as the features
        img = np.insert(img, 0, label_num)
        # Add the data to the dataframe
        dataset = dataset.append(pd.Series(img, index=dataset.columns), ignore_index=True)
        print(img.shape, label_num)

# Save the dataset as a csv file
dataset.to_csv(os.path.join(csv_path), index=False)
print('Dataset saved as csv file')