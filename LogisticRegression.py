# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = load_breast_cancer() # 导入乳腺癌数据集
X = data.data
y = data.target
#data.data.shape
# 建立两个逻辑回归，solver 参数代表的时候选取何种算法进行优化
lrl1 = LR(penalty="l1",solver="liblinear",C=0.5,max_iter=1000)
lrl2 = LR(penalty="l2",solver="liblinear",C=0.5, max_iter=1000)
#逻辑回归的重要属性coef_，查看每个特征所对应的参数
lrl1 = lrl1.fit(X,y)
print ("lrl1.coef_=",lrl1.coef_)
(lrl1.coef_ != 0).sum(axis=1)
lrl2 = lrl2.fit(X,y)
print ("lrl2.coef_=",lrl2.coef_)
#画图，测试不同的参数
l1 = []
l2 = []
l1test = []
l2test = []
#分割输数据及
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
for i in np.linspace(0.05,1,19): # 选取不同的 C值 实验，正则化强度的倒数
    lrl1= LR(penalty="l1",solver="liblinear",C=i,max_iter=1000)
    lrl2= LR(penalty="l2",solver="liblinear",C=i,max_iter=1000)
    
    lrl1 = lrl1.fit(Xtrain,Ytrain)
    l1.append(accuracy_score(lrl1.predict(Xtrain),Ytrain))
    l1test.append(accuracy_score(lrl1.predict(Xtest),Ytest))
    
    lrl2 = lrl2.fit(Xtrain, Ytrain)
    l2.append(accuracy_score(lrl2.predict(Xtrain), Ytrain))
    l2test.append(accuracy_score(lrl2.predict(Xtest), Ytest))
graph = [l1, l2, l1test, l2test]
color = ["green", "black", "lightgreen", "gray"]
label = ["L1", "L2", "L1test", "L2test"]
plt.figure(figsize=(6, 6))
for i in range(len(graph)):
    plt.plot(np.linspace(0.05, 1, 19), graph[i], color[i], label=label[i])
plt.legend(loc=4) # 图例的位置在哪里?4表示，右下角
plt.show()