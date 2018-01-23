import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR


print('Generate dataset')


def poly_noise(x, noise=0):
    a, b, c = 2, 1, 4
    return a + b * x + c * x ** 2 + noise * np.random.randn()

x = 2 * np.random.random(100) - 1
X = list(map(lambda x: [x], x))
y = list(map(lambda x: poly_noise(x, noise=0.5), x))

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
