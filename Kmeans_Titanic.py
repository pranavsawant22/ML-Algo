import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np

df = pd.read_excel("titanic.xls")
df.drop(['name', 'body'], 1, inplace=True)
df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)


def handle_non_numerics_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)

            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))
    return df


df = handle_non_numerics_data(df)
df.drop(['sibsp'],1,inplace=True)
X = (np.array(df.drop(['survived'],1).astype(float)))
X = preprocessing.scale(X)
Y = np.array(df['survived'])
clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    predictiction = clf.predict(predict_me)
    if predictiction[0] == Y[i]:
        correct+=1

print(correct/len(X))