import os
import sys

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utils'))
import my_plots


print('Load data')
iris = load_iris()
X = iris.data[:, 2:]  # petal length and witdh
y = iris.target

print('Train decision tree classifier')
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

print('Display tree structure')

print('feature_names', iris.feature_names[2:])
print('target_names', iris.target_names)

export_graphviz(
    tree_clf,
    out_file='iris_tree.dot',
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True,
    )
os.system('dot -Tpng iris_tree.dot -o iris_tree.png')

print('Get probabilities')
probs = tree_clf.predict_proba([[5, 1.5]])
print('Probabilities', probs)
result = tree_clf.predict([[5, 1.5]])
print('Result', result)

print('Visualize decision boundaries')

my_plots.plot_clf_train_scatter(X, y, tree_clf,
    x_label=iris.feature_names[2],
    y_label=iris.feature_names[3],
    save_name='decision_tree_decision_boundaries.png')
