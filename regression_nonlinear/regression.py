import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utils'))
import my_plots

print('Generate dataset')


def poly_noise(x, noise=0):
    a, b, c = 2, 1, 4
    return a + b * x + c * x ** 2 + noise * np.random.randn()

x = 2 * np.random.random(100) - 1
X = np.asarray(list(map(lambda x: [x], x)))
y = np.asarray(list(map(lambda x: [poly_noise(x, noise=0.5)], x)))

print('Plot it')
fig = plt.figure()
plt.title('Polynomial dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x, y)
plt.savefig('poly_scatter.pdf')

print('Define the model')
svm_poly_reg = SVR(kernel='poly', degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)

print('Prediction', svm_poly_reg.predict([[0.5]]))
print('Parameters', svm_poly_reg.get_params())

print('Visualize training')
my_plots.plot_reg_train_scatter(X, y, svm_poly_reg)
