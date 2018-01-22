import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


iris = datasets.load_iris()
print(type(iris))
print(iris.keys())
print(type(iris.data))
print(iris.data[0])
print(iris.data[0, (2, 3)])

X = iris['data'][:, (2, 3)]  # petal length and width
y = (iris['target'] == 2).astype(np.float64)  # Iris-Virginica

svm_clf = Pipeline((
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C=1, loss='hinge')),
    # ('linear_svc', SVC(kernel='linear', C=1),
    ))

svm_clf.fit(X, y)

result = svm_clf.predict([[5.5, 1.7]])
print('Result:', result)
