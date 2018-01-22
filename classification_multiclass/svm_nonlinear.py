import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

print('Generate toy data')
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

print('Plot it')
fig = plt.figure()
plt.title('Moon dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(list(map(lambda x: x[0], X)), list(map(lambda x: x[1], X)), c=y)
plt.savefig('moon_scatter.pdf')

print('Define model')
polynomial_svm_clf = Pipeline((
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C=10, loss='hinge')),
    ))

print('Train model')
polynomial_svm_clf.fit(X, y)

print('Apply model')
X_test, y_test = make_moons(n_samples=1, noise=None, random_state=42)
result = polynomial_svm_clf.predict(X_test)
print('Predicted', result, 'Real', y_test)
