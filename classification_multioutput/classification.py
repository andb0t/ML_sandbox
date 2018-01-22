# Analysis goal: remove noise from image

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


print('Multioutput classification')

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test


from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)

print('Test it')
some_index = 200
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
plt.savefig('clean_digit.pdf')
plot_digit(y_test_mod[some_index])
plt.savefig('orig_digit.pdf')
