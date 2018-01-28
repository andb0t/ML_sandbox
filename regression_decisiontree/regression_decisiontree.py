import os
import sys

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utils'))
import my_plots


print('Generate dataset')


def poly_noise(x, noise=0):
    a, b, c = 2, 1, 4
    return a + b * x + c * x ** 2 + noise * np.random.randn()

x = 2 * np.random.random(100) - 1
X = list(map(lambda x: [x], x))
y = list(map(lambda x: poly_noise(x, noise=0.5), x))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

print('Define and train decision tree regressors')
tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_train, y_train)

gbr_strategy = 'early_stop_incremental'
if gbr_strategy == 'single':
    gbrt_reg = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
    gbrt_reg.fit(X_train, y_train)
elif gbr_strategy == 'early_stop':
    gbrt_stop_reg = GradientBoostingRegressor(max_depth=2, n_estimators=120)
    gbrt_stop_reg.fit(X_train, y_train)
    errors = [mean_squared_error(y_val, y_pred)
              for y_pred in gbrt_stop_reg.staged_predict(X_val)]
    best_n_estimators = np.argmin(errors)
    gbrt_reg = GradientBoostingRegressor(max_depth=2,
                                         n_estimators=best_n_estimators)
    gbrt_reg.fit(X_train, y_train)
elif gbr_strategy == 'early_stop_incremental':
    gbrt_reg = GradientBoostingRegressor(max_depth=2, warm_start=True)
    min_val_error = float('inf')
    error_going_up = 0
    for n_estimators in range(0, 120):
        gbrt_reg.estimators = n_estimators
        gbrt_reg.fit(X_train, y_train)
        y_pred = gbrt_reg.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:
                break  # early stopping
elif gbr_strategy == 'stacking':
    print('Split up training sample further')
    X_layer, X_blend, y_layer, y_blend = train_test_split(
        X_train, y_train, test_size=0.5)
    X_layer_train, X_layer_val, y_layer_train, y_layer_val = train_test_split(
        X_layer, y_layer, test_size=0.25)
    X_blend_train, X_blend_val, y_blend_train, y_blend_val = train_test_split(
        X_blend, y_blend, test_size=0.25)
    print('Train regular forest layer')
    print('Train blender layer')
    # TODO: implement stacking


print('Visualize training')
for reg in (tree_reg, gbrt_reg):
    name = reg.__class__.__name__
    my_plots.plot_reg_train_scatter(X_train, y_train, reg,
                                    title=name,
                                    save_name='reg_model_' + name + '.png')

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
