from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import glob

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

images_per_class = 50
fixed_size       = tuple((500, 500))
class_pre = 'daisy'
train_path       = "dataset\\train\\flower"
test_path       = "dataset\\test\\flower\\"+class_pre
test_size = 0.20
seed      = 9
test_image = 'daisy.jpg'
# get the training labels
train_labels = os.listdir(train_path)
bins    = 8

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_images = []
labels          = []
test_images = []
test_lables = []

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
            histogram  = fd_histogram(image)
            # print(image.shape)
        except Exception as e:
            print(file)
            print(str(e))
        # image_reshape = np.array(haralick).reshape(reshape_size)

        # pyplot.imshow(haralick)
        # pyplot.show()

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_images.append(np.hstack([haralick,histogram]))

# test folder
for test_file in glob.glob(test_path + "/*.jpg"):
    try:
        file_name = test_file.replace(test_path + '\\', '')
        dir_test_file = dir + "\\"+file_name
        image = cv2.imread(dir_test_file)
        image = cv2.resize(image, fixed_size)
        haralick = fd_haralick(image)
        histogram = fd_histogram(image)
    except Exception as e:
        print(file)
        print(str(e))
    test_lables.append(file_name)
    test_images.append(np.hstack([haralick,histogram]))

a = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(np.array(global_images),
                                                    np.array(labels),
                                                    test_size=test_size)

models = []
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=9)))
models.append(('KNN', KNeighborsClassifier()))

print(X_train.shape)
# X_t_train = pca.fit_transform(X_train)
# X_t_test = pca.fit_transform(X_test)

for name ,model in models:
    model.fit(X_train, y_train)
    # use the model to predict the labels of the test data
    model_score = model.score(X_test, y_test)
    predicted = model.predict(np.array(X_test))
    expected = np.array(y_test)
    # print(predicted)
    # print(expected)
    predicted = model.predict(np.array(test_images))
    expected = np.array(test_lables)
    count = 0
    print(predicted)
    for i in predicted:
        if i == class_pre:
            count = count + 1
    #
    print(expected)
    msg = "%s: %f , act = %f" % (name,model_score,count/len(expected))
    print(msg)
