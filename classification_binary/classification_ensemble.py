import os
import sys

from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utils'))
import my_plots


print('Generate toy data')
X, y = make_moons(n_samples=5000, noise=0.2, random_state=42)

print('Split into training and test sample')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Define classifiers')
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
svm_clf_with_prob = SVC(probability=True)

voting_strat = 'soft'
if voting_strat == 'hard':
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard')
if voting_strat == 'soft':
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf_with_prob)],
        voting='soft')

sampling_strat = 'bagging'
if sampling_strat == 'bagging':
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1,
        oob_score=True)
elif sampling_strat == 'pasting':
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=False, n_jobs=-1)

tree_clf = DecisionTreeClassifier()
ext_clf = ExtraTreesClassifier()
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm='SAMME.R', learning_rate=0.5)

print('Train all of them and check their accuracy')
for clf in (log_clf, rnd_clf, svm_clf, voting_clf, bag_clf, tree_clf, ext_clf, ada_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    name = clf.__class__.__name__
    print(name, accuracy_score(y_test, y_pred))
    my_plots.plot_clf_train_scatter(
        X_test, y_test, clf,
        title=name,
        save_name='clf_model_' + name + '.png')

if sampling_strat == 'bagging':
    print('Evaluate bagging clf')
    print('Out-of-bag score:', bag_clf.oob_score_)
    y_pred = bag_clf.predict(X_test)
    print('Compared to actual score:', accuracy_score(y_test, y_pred))
