# Analysis goal: differentiate MNIST 5s from non-5s

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


def plot_roc_curve(fpr, tpr, label=None):
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.savefig('precision_recall_threshold_curve.pdf')

plot_precision_vs_recall(precisions, recalls)
plt.savefig('precision_recall_curve.pdf')
#

from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
plot_roc_curve(fpr, tpr)
plt.savefig('roc_curve.pdf')

auc = roc_auc_score(y_train_5, y_scores)
print('AUC', auc)

print('Try random forest')

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


def plot_roc_SGD_vs_forest(fpr, tpr, fpr_forest, tpr_forest):
    plt.figure()
    plt.plot(fpr, tpr, 'b:', label='SGD')
    plt.plot(fpr_forest, tpr_forest, label='Random Forest')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

plot_roc_SGD_vs_forest(fpr, tpr, fpr_forest, tpr_forest)
plt.savefig('roc_curve_SGD_vs_forest.pdf')

auc_forest = roc_auc_score(y_train_5, y_scores_forest)
print('AUC forest', auc)
