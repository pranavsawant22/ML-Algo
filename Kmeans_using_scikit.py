import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import style
style.use("ggplot")

X = np.array([[1,2],[4,0.6],[5,7],[9,11],[7,8],[3,1.8]])

# plt.scatter(X[:,0],X[:,1])
# plt.show()

clf = KMeans(n_clusters=1)
clf.fit(X)
 
centroids = clf.cluster_centers_
labels = clf.labels_

colors = ['r.','g.','b.','k.','c.'] * 10

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize = 25)
plt.scatter(centroids[:,0],centroids[:,1],marker="x",linewidths=5,s=150)
plt.show()