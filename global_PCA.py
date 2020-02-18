from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py



images_per_class = 80
fixed_size       = tuple((500, 500))
h5_data          = 'output\\data_pca.h5'
h5_labels        = 'output\\labels_pca.h5'
train_path       = "dataset\\train\\flower"


# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels          = []

for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    file_list = []
    for x in range(1,images_per_class+1):