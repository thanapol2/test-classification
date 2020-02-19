from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

images_per_class = 30
fixed_size       = tuple((500, 500))
h5_data          = 'output\\data_pca.h5'
h5_labels        = 'output\\labels_pca.h5'
train_path       = "dataset\\train\\test"
reshape_size  = tuple((500,-1))
test_size = 0.20
seed      = 9
test_image = 'daisy.jpg'
# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_images = []
labels          = []


pca = PCA(n_components=2)
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    file_list = []
    for x in range(1,images_per_class+1):
        try:
            file = dir + "\\" + str(x) + ".jpg"
            image = cv2.imread(file)
            image = cv2.resize(image, fixed_size)
            haralick = fd_haralick(image)
            # print(image.shape)
        except Exception as e:
            print(file)
            print(str(e))
        # image_reshape = np.array(haralick).reshape(reshape_size)

        # pyplot.imshow(haralick)
        # pyplot.show()

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_images.append(haralick)

# proj = pca.fit_transform(digits.data)
# pyplot.scatter(proj[:, 0], proj[:, 1], c=digits.target)
#
# pyplot.colorbar()
#
# pyplot.show()

X_train, X_test, y_train, y_test = train_test_split(np.array(global_images),
                                                    np.array(labels),
                                                    test_size=test_size,
                                                    random_state=seed)

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=9)))

print(X_train.shape)
# X_t_train = pca.fit_transform(X_train)
# X_t_test = pca.fit_transform(X_test)

for name ,model in models:
    model.fit(X_train, y_train)
    # use the model to predict the labels of the test data
    predicted = model.predict(X_test)
    expected = y_test

    model_score = model.score(X_test, y_test)
    print(predicted)
    #
    print(expected)
    msg = "%s: %f " % (name,model_score)
    print(msg)
