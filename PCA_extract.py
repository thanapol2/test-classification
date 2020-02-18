from sklearn.decomposition import PCA
import cv2
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
import h5py

image_file = "test.jpg"
pca = PCA(n_components=2)
image = cv2.imread(image_file)

image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# pyplot.imshow(image)
# pyplot.show()
a = np.array(image).reshape(480,-1)
print(a)
# pyplot.imshow(a)
# pyplot.show()
proj = pca.fit_transform(a.data)



targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))