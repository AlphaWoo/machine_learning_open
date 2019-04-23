# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
X, y = make_blobs(n_samples=500,n_features=2,centers=5,random_state=1)
fig, ax1 = plt.subplots(1)
ax1.scatter(X[:, 0],X[:, 1], marker='o' ,s=8)
n_clusters = 5
cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(X)
y_pred = cluster.labels_
centroid = cluster.cluster_centers_
color = ["red","pink","orange","gray","green"]
fig.ax1 = plt.subplots(1)
for i in range (n_clusters):
    ax1.scatter(X[y_pred == i,0],X[y_pred == i,1],marker='o',s=8,c=color[i])
ax1.scatter(centroid[:,0],centroid[:,1],marker='o',s=40,c="black")
#plt.show()