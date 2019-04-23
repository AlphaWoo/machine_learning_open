# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import time
img = Image.open('3.jpg')
img.show()

def dist_euc(X, Y):
    n, d = np.shape(X)[0], np.shape(X)[1]
    m, d = np.shape(Y)[0], np.shape(Y)[1]
    # for i in range(n):
    #     for j in range(m):
    #         d = np.sqrt(np.sum((X[i] - Y[j])**2))
    diff = np.tile(X[:,np.newaxis,:], (1,m,1)) - np.tile(Y[np.newaxis,:,:], (n,1,1))
    dist = np.sqrt(np.sum(np.square(diff), axis=-1, keepdims=False))
    return dist

data = np.array(img,dtype=float)
shape = np.shape(data)
w,h,c=shape[0],shape[1],shape[2]
data = np.reshape(data,newshape=(w*h,c))
mask = np.zeros(shape=[w*h,c],dtype=np.uint8)

k=5
num=w*h
Iteration = 20
idx=[int(np.random.rand(1)*num) for i in range(k)]
centers = data[idx]
labels = np.zeros(shape=(num),dtype=int)
colors = [[255,182,193],[30,144,255],[0,255,255],[255,20,147],[0,255,0],[255,140,0],[255,69,0],[0,0,0]]
np.random.seed(int(time.time())%1000)
for it in range(Iteration):
    dist = dist_euc(data,centers)
    labels=np.argmin(dist,axis=-1)
    for i in range(k):
        idx = [j for j in range(num) if labels[j] == i]
        centers[i]=np.mean(data[idx],axis=0)
        mask[idx] = colors[i]
    new_im=Image.fromarray(np.reshape(mask,newshape=[w,h,c]))
    if it%1 == 0:
        new_im.show()
        