# -*- coding: utf-8 -*-
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
X,y = make_blobs(n_samples=500,n_features=2,centers=4,random_state=1)
fig,ax1 = plt.subplots(1)
ax1.scatter(X[:,0],X[:,1],marker='o',color='m',s=8)
plt.show()

color = ["red","pink","orange","gray"]
fig,ax1 = plt.subplots(1)

for i in range(4):
    ax1.scatter(X[y==i,0],X[y==i,1],marker='o',s=8,c=color[i])
plt.show()