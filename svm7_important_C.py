from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime
data = load_breast_cancer()
X = data.data

y = data.target
import pandas as pd
data = pd.DataFrame(X)

# print(data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)
#统一量刚单位
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
data = pd.DataFrame(X)
data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)


#调线性核函数
score = []
C_range = np.linspace(0.01,30,50)
for i in C_range:
    clf = SVC(kernel="linear",C=i,cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))
print("max(score), C_range[score.index(max(score))]",max(score), C_range[score.index(max(score))])
plt.plot(C_range,score)
plt.show()
#更换核函数，变为rbf
score = []
C_range = np.linspace(0.01,30,50)
for i in C_range:
    clf = SVC(kernel="rbf",C=i,gamma =0.012742749857031322,cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))
print("max(score), C_range[score.index(max(score))]",max(score), C_range[score.index(max(score))])
plt.plot(C_range,score)
plt.show()
#进一步细化,依然采用rbf, 但是C的范围缩小到 【5，7】
score = []
C_range = np.linspace(5,7,50)
for i in C_range:
    clf = SVC(kernel="rbf",C=i,gamma =0.012742749857031322,cache_size=5000).fit(Xtrain,Ytrain)
    print("C_range",i)
    score.append(clf.score(Xtest,Ytest))
print("max(score), C_range[score.index(max(score))]",max(score), C_range[score.index(max(score))])
plt.plot(C_range,score)
plt.show()