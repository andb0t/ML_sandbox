import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utils'))
import my_plots

print('Generate dataset')


def poly_noise(x, noise=0):
    a, b, c = 2, 1, 4
    return a + b * x + c * x ** 2 + noise * np.random.randn()

x = 2 * np.random.random(100) - 1
X = list(map(lambda x: [x], x))
y = list(map(lambda x: [poly_noise(x, noise=0.5)], x))

print('Plot it')
fig = plt.figure()
plt.title('Polynomial dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x, y)
plt.savefig('poly_scatter.pdf')

print('Define the model')

model = 'linear_with_poly'

if model == 'SVR':
    poly_reg = SVR(kernel='poly', degree=2, C=100, epsilon=0.1)
    poly_reg.fit(X, y)

elif model == 'linear_with_poly':
    from sklearn.pipeline import Pipeline
    poly_reg = Pipeline([
        ('addpoly', PolynomialFeatures(degree=2, include_bias=False)),
        ('lin_reg', LinearRegression())
        ])
    poly_reg.fit(X, y)

print('Prediction', poly_reg.predict([[0.5]]))
print('Parameters', poly_reg.get_params())

print('Visualize training')
my_plots.plot_reg_train_scatter(X, y, poly_reg)
