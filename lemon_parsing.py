# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 03:24:14 2020

@author: Alice

to use on the lemon dataset: https://github.com/softwaremill/lemon-dataset/
"""

import json
import pandas as pd
import os
import cv2
from fruit_classification import *

dim = 100

with open(r"./lemon-dataset/annotations/instances_default.json", "r") as read_file:
    data = json.load(read_file)

images_df = pd.DataFrame(data['images'])
annotations_df = pd.DataFrame(data['annotations'])
df = pd.concat([images_df, annotations_df])
df = df[["id", "file_name", "category_id". "image_id"]]
df = df[~df["file_name"].isna()]  # remove rows in dataframe with nan filename
df["file_name"] = df["file_name"].apply(lambda i: os.path.normpath(i))

# get images and labels
images_training = []
labels_training = []
images_testing = []
labels_testing = []
path = ".\lemon-dataset\images\\"
image_type_count = 0 # sloppy solution to separate out training/test sets

for image_path in glob.glob(os.path.join(path, "*.jpg")):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (dim, dim))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    label = df.loc[df["file_name"] == image_path.split('\\', 2)[-1], "category_id"].isna()
    if image_type_count == 2:
        images_testing.append(image)
        labels_testing.append(label.iloc[0])
        image_type_count = 0
    else:
        images_training.append(image)
        labels_training.append(label.iloc[0])
        image_type_count += 1

    
#print("There are " , len(images_training) , " " , data_type.upper(), " images of " , fruits[i].upper())
#images = np.array(images)
#labels = np.array(labels)
