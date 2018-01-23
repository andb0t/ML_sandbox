import os
import sys

import numpy as np
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utils'))
import my_plots

iris = datasets.load_iris()
print(type(iris))
print(iris.keys())
print(type(iris.data))
print(iris.data[0])
print(iris.data[0, (2, 3)])

X = iris['data'][:, (2, 3)]  # petal length and width
y = (iris['target'] == 2).astype(np.float64)  # Iris-Virginica

SVM_algo = 'SGD_SVC'

if SVM_algo == 'linear_optimized':
    svm_clf = Pipeline((
        ('scaler', StandardScaler()),
        ('linear_svc', LinearSVC(C=1, loss='hinge')),
        # ('linear_svc', SVC(kernel='linear', C=1),
        ))
elif SVM_algo == 'linear':
    svm_clf = Pipeline((
        ('scaler', StandardScaler()),
        ('linear_svc', SVC(kernel='linear', C=10)),
        ))
elif SVM_algo == 'SGD_SVC':
    svm_clf = Pipeline((
        ('scaler', StandardScaler()),
        ('linear_svc', SGDClassifier(loss='hinge')),  # hinge loss results in SVM
        ))

svm_clf.fit(X, y)

result = svm_clf.predict([[5.5, 1.7]])
print('Result:', result)

print('Visualize decision boundaries')

my_plots.plot_clf_train_scatter(X, y, svm_clf,
    x_label=iris.feature_names[2],
    y_label=iris.feature_names[3],
    save_name='svm_linear_decision_boundaries.png'
    )
