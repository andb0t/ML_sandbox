# Analysis goal: decide if MNIST picture is an odd number >= 7

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Download the data
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
print(mnist)
X, y = mnist['data'], mnist['target']
print(X.shape)
print(y.shape)

# Inspect one instance
some_digit = X[36000]


def plot_digit(data):
    plt.figure()
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")

plot_digit(some_digit)
plt.savefig('example_instance.pdf')

# Get the test data
MNIST_TRAIN_DIV = 60000
X_train, X_test, y_train, y_test = X[:MNIST_TRAIN_DIV], X[MNIST_TRAIN_DIV:], y[:MNIST_TRAIN_DIV], y[MNIST_TRAIN_DIV:]
shuffle_index = np.random.permutation(MNIST_TRAIN_DIV)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

print('Train multilabel classifier:')

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_train_multilabel = np.c_[y_train_large, y_train_odd]

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train_multilabel)
result = knn_clf.predict([some_digit])
print(result)

print('Evaluate the model')
from sklearn.model_selection import cross_val_predict
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
from sklearn.metrics import f1_score
print(f1_score(y_train, y_train_knn_pred, average='macro'))  # macro='weighted' for more weight to more aboundant classes
