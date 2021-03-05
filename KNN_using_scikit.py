import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_validate

import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data')

df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)
x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)

print(accuracy * 100)

examples_measure = np.array([[8, 2, 1, 1, 1, 2, 3, 2, 1], [1, 7, 7, 6, 4, 10, 4, 1, 5]])
examples_measure = examples_measure.reshape(len(examples_measure), -1)
print(clf.predict(examples_measure), len(examples_measure))
