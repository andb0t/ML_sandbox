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

print('Evaluate the classifiers using cross-validation')
from sklearn.model_selection import cross_val_score
cvs = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')
print('Cross-validation score', cvs)

print('Scale input to increase accuracy')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cvs_scaled = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')
print('Cross-validation score scaled training data', cvs_scaled)

print('Error analysis')
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
from sklearn.metrics import confusion_matrix
conf_mx = confusion_matrix(y_train, y_train_pred)
print('Confusion matrix', conf_mx)

print('Normalize confusion matrix')
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
print('Scaled confusion matrix', norm_conf_mx)

plt.figure()
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.savefig('confusion_matrix.pdf')

print('Plot some of the often falsely classified images')


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axis("off")

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8, 8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.savefig('errors_visualized.pdf')
