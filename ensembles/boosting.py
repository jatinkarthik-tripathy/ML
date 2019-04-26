import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

#data acquisition and preproc
iris_data = load_iris()
X = iris_data.data
Y = iris_data.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#GradBoost implemented on Decision Trees
trees = 100
clf = GradientBoostingClassifier(n_estimators=trees)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(f'GradBoost Accuracy: {accuracy}')


#AdaBoost implemented on Decision Trees
clf = AdaBoostClassifier(n_estimators=trees)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(f'AdaBoost Accuracy: {accuracy}')
