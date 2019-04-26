import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

#data acquisition and preproc
iris_data = load_iris()
X = iris_data.data
Y = iris_data.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#Bagging implemented on Decision Trees
trees = 100
dec_tree = DecisionTreeClassifier()
clf = BaggingClassifier(base_estimator=dec_tree , n_estimators=trees)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(f'Bagging Accuracy: {accuracy}')


#Random Forest
clf = RandomForestClassifier(n_estimators=trees)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(f'Random Forest Accuracy: {accuracy}')
