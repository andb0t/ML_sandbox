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


def show_image(some_digit_image):
    plt.figure()
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.savefig('example_instance.pdf')

some_digit_image = some_digit.reshape(28, 28)
show_image(some_digit_image)

# Get the test data
MNIST_TRAIN_DIV = 60000
X_train, X_test, y_train, y_test = X[:MNIST_TRAIN_DIV], X[MNIST_TRAIN_DIV:], y[:MNIST_TRAIN_DIV], y[MNIST_TRAIN_DIV:]
shuffle_index = np.random.permutation(MNIST_TRAIN_DIV)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

print('Train multiclass classifier:')

from sklearn.linear_model import SGDClassifier
#
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)  # automatic OvA strategy to make SGD multiclass
result = sgd_clf.predict([some_digit])
print(result)

print('Check decision function')

some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)
max_pos = np.argmax(some_digit_scores)
print(max_pos)
print(sgd_clf.classes_)

print('The highest score belongs to class', sgd_clf.classes_[max_pos])

print('Force OvO strategy')

from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
result = ovo_clf.predict([some_digit])
print('Result OvO SGD', result)
print(len(ovo_clf.estimators_))

print('Use random forest')
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
# random forests are multiclass classifiers, so no OvO or OvA necessary
forest_clf.fit(X_train, y_train)
result_forest = forest_clf.predict([some_digit])
print('Result random forest', result_forest)
print('Probabilities', forest_clf.predict_proba([some_digit]))
