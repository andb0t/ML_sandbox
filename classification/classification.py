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
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.savefig('test.pdf')

# Get the test data
MNIST_TRAIN_DIV = 60000
X_train, X_test, y_train, y_test = X[:MNIST_TRAIN_DIV], X[MNIST_TRAIN_DIV:], y[:MNIST_TRAIN_DIV], y[MNIST_TRAIN_DIV:]
shuffle_index = np.random.permutation(MNIST_TRAIN_DIV)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

print('The types and shapes of the train variables')
print(type(X_train), X_train.shape)
print(type(y_train), y_train.shape)

# Train for classification of 5 and non-5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=1337)
sgd_clf.fit(X_train, y_train_5)
result = sgd_clf.predict([some_digit])
print(result)
