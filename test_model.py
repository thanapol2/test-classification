
# to visualize results
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import mahotas
# import global_py as gl

#--------------------
# tunable-parameters
#--------------------
num_trees = 100
test_path  = "dataset/test"
fixed_size = tuple((500, 500))
seed      = 9
bins             = 8
global_feature = []



def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

def train(model,train_labels):
    count = 0
    test_results = []
    # loop through the test images
    for file in glob.glob(test_path + "/*.jpg"):
        file_name = file.replace(test_path+'\\','')
        # read the image
        image = cv2.imread(file)

        # resize the image
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature.append(np.hstack([fv_histogram, fv_haralick, fv_hu_moments]))

        # scale features in the range (0-1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_feature = scaler.fit_transform(global_feature)

        # predict label of test image
        prediction = model.predict(rescaled_feature)[0]
        test_results.append([file_name,train_labels[prediction]])
    #     if(train_labels[prediction]==test_name):
    #         count = count +1
    #
    # print ("result %s : %d" %(test_name,count))
    #     show predicted label on image
    #     cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
    #
    #     # display the output image
    #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     plt.show()
    return (test_results)


# def train(model,train_labels,test_name,dir):
#     count = 0
#     test_results = []
#     # loop through the test images
#     for file in glob.glob(dir + "/*.jpg"):
#         file_name = file.replace(dir,'')
#         # read the image
#         image = cv2.imread(file)
#
#         # resize the image
#         image = cv2.resize(image, fixed_size)
#
#         ####################################
#         # Global Feature extraction
#         ####################################
#         fv_hu_moments = fd_hu_moments(image)
#         fv_haralick   = fd_haralick(image)
#         fv_histogram  = fd_histogram(image)
#
#         ###################################
#         # Concatenate global features
#         ###################################
#         global_feature.append(np.hstack([fv_histogram, fv_haralick, fv_hu_moments]))
#
#         # scale features in the range (0-1)
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         rescaled_feature = scaler.fit_transform(global_feature)
#
#         # predict label of test image
#         prediction = model.predict(rescaled_feature)[0]
#         test_results.append([file_name,train_labels[prediction]])
#         if(train_labels[prediction]==test_name):
#             count = count +1
#
#     print ("result %s : %d" %(test_name,count))
#         # show predicted label on image
#         # cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
#         #
#         # # display the output image
#         # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         # plt.show()
#     return (test_results)