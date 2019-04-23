# -*- coding: utf-8 -*-
from numpy import *
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
with open('D:/Machine Learning/2 Clustering_Algrithm_20190224/data1.pkl', 'rb') as file1:
    data1 = pickle.load(file1)
#print(data1)
labels='Class1','Class2','Class3'
colors='yellow','gold','lights'
file1.close()
#print("data的长度：",data1.ndim)
plt.scatter(data1[:,0],data1[:,1],color='b',s=2)
plt.show()
A1,A2,A3=[2,0],[8,5],[2,10]
#print(A1,A2,A3)
#for DATA in data1:
dist1=numpy.sqrt(numpy.sum(numpy.square(data1-A1)))
dist2=numpy.sqrt(numpy.sum(numpy.square(data1-A2)))
dist3=numpy.sqrt(numpy.sum(numpy.square(data1-A3)))
print(dist1)
print(dist2)
print(dist3)



