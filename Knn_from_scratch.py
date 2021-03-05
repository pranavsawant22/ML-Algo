# Video Link - https://www.youtube.com/watch?v=hl3bQySs8sM&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=16
import random

import numpy as np
import matplotlib.pyplot as plt
import warnings
from collections import Counter
from matplotlib import style
import pandas as pd

style.use('fivethirtyeight')


# dataset = {'k': [[1, 2], [2, 3], [3, 1], [4, 2], [1, 4]], 'r': [[5, 7], [6, 5], [8, 6], [7, 9], [9, 8]]}
# new_features = [2, 4]

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1])
# plt.show()


def k_nearest_neighbours(data, predict, k=3):
    if k < len(data):
        warnings.warn("K is set to value less than total groups")
    distances = []
    for groups in data:
        for features in data[groups]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, groups])
    # print(sorted(distances))
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


# result = k_nearest_neighbours(dataset, new_features, k=3)
# print("Nearest cluster is : " + result)

df = pd.read_csv("breast-cancer-wisconsin.data")
df.drop(['id'], 1, inplace=True)
df.replace('?', -99999, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

print(' # ' * 20)
test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])
total, correct = 0, 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbours(train_set, data, k=101)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct / total)
