import os

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz


print('Generate dataset')


def poly_noise(x, noise=0):
    a, b, c = 2, 1, 4
    return a + b * x + c * x ** 2 + noise * np.random.randn()

x = 2 * np.random.random(100) - 1
X = list(map(lambda x: [x], x))
y = list(map(lambda x: poly_noise(x, noise=0.5), x))

print('Define and train decision tree regressor')
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)

print('Display decision tree')
export_graphviz(
    tree_reg,
    out_file='poly_tree.dot',
    # feature_names=['y'],
    rounded=True,
    filled=True,
    )
os.system('dot -Tpng poly_tree.dot -o poly_tree.png')

print('Use model')
print('(predicted value = average of all training instances in leaf)')
result = tree_reg.predict([[0.6]])
print('Result', result)
