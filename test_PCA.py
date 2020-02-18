from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()

# show hand write digi
# fig = pyplot.figure(figsize=(8, 8))
# for i in range(64):
#     ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
#     ax.imshow(digits.images[i], cmap=pyplot.cm.binary, interpolation='nearest')
# pyplot.show()
print(digits.data.shape)
pca = PCA(n_components=2)
proj = pca.fit_transform(digits.data)
pyplot.scatter(proj[:, 0], proj[:, 1], c=digits.target)

pyplot.colorbar()

pyplot.show()



# split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

# train the model
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=9)))

X_t_train = pca.transform(X_train)
X_t_test = pca.transform(X_test)

for name ,model in models:
    model.fit(X_t_train, y_train)
    # use the model to predict the labels of the test data
    predicted = model.predict(X_t_test)
    expected = y_test

    model_score = model.score(X_t_test, y_test)
    # print(predicted)
    #
    # print(expected)
    msg = "%s: %f " % (name,model_score)
    print(msg)
