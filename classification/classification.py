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

print('The types and shapes of the train variables')
print(type(X_train), X_train.shape)
print(type(y_train), y_train.shape)

# Train for classification of 5 and non-5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier
#
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
result = sgd_clf.predict([some_digit])
print(result)

print('Cross validation:')

from sklearn.model_selection import cross_val_score
cvs = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print(cvs)

print('Confusion matrix:')

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
print('confusion_matrix:', confusion_matrix(y_train_5, y_train_pred))
print('precision_score:', precision_score(y_train_5, y_train_pred))
print('recall_score:', recall_score(y_train_5, y_train_pred))
print('f1_score:', f1_score(y_train_5, y_train_pred))

print('Get decision scores:')
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(prec, rec, thresh):
    plt.figure()
    plt.plot(thresh, prec[:-1], 'b--', label='Precision')
    plt.plot(thresh, rec[:-1], 'g--', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0, 1])


def plot_precision_vs_recall(prec, rec):
    plt.figure()
    plt.plot(rec, prec, 'b--', label='Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.savefig('precision_recall_threshold_curve.pdf')

plot_precision_vs_recall(precisions, recalls)
plt.savefig('precision_recall_curve.pdf')
