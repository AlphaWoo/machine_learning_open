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
print ('data.shape:{0};no.positive:{1};no.negative:{2}'.format(X.shape,y[y==1].shape[0],y[y==0].shape[0])) #重点分析

#print('data.shape:{0};no. positive:{1};no. negative:{2}'.format(X.shape, y[y == 1].shape[0], y[y == 0].shape[0]))
print(data.data[0])
#Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.2)
model = LR()
model.fit(Xtrain,Ytrain)
y_pred = model.predict(Xtest)
print ('测试集上标签预测正确的个数是{}；测试集标签总数是：{}'.format(np.equal(y_pred,Ytest).sum(),Ytest.shape[0]))

#data.data.shape
# 建立两个逻辑回归，solver 参数代表的时候选取何种算法进行优化
#lrl1 = LR(penalty="l1",solver="liblinear",C=0.5,max_iter=1000)
#lrl2 = LR(penalty="l2",solver="liblinear",C=0.5, max_iter=1000)
#逻辑回归的重要属性coef_，查看每个特征所对应的参数
#lrl1 = lrl1.fit(X,y)
#print ("lrl1.coef_=",lrl1.coef_)
#(lrl1.coef_ != 0).sum(axis=1)
#lrl2 = lrl2.fit(X,y)
#print ("lrl2.coef_=",lrl2.coef_)
#画图，测试不同的参数
#l1 = []
#l2 = []
#l1test = []
#l2test = []
#分割输数据及
#Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)

#model = LR()
#model.fit(Xtrain,Ytrain)
#
#y_pred = model.predict(Xtest)
#print ('测试集上标签预测正确的个数是{}；测试集标签总数是：{}'.format(np.equal(y_pred,Ytest).sum(),Ytest.shape[0]))
train_score = model.score(Xtrain,Ytrain)
test_score = model.score(Xtest,Ytest)
print('train score:{train_score:.6f};test score:{test_score:.6f}'.format(train_score=train_score,test_score=test_score))

#for i in np.linspace(0.05,1,19): # 选取不同的 C值 实验，正则化强度的倒数
#    lrl1= LR(penalty="l1",solver="liblinear",C=i,max_iter=1000)
##    lrl2= LR(penalty="l2",solver="liblinear",C=i,max_iter=1000)
#    
#    lrl1 = lrl1.fit(Xtrain,Ytrain)
#    l1.append(accuracy_score(lrl1.predict(Xtrain),Ytrain))
#    l1test.append(accuracy_score(lrl1.predict(Xtest),Ytest))
#
#print("测试集上标签预测正确的个数是：",np.size(l1))
#print("测试集上标签总数是：",np.size(X))
#print("train score：",accuracy_score(lrl1.predict(Xtrain),Ytrain))
#print("test score：",accuracy_score(lrl1.predict(Xtest),Ytest))
    
#    lrl2 = lrl2.fit(Xtrain, Ytrain)
#    l2.append(accuracy_score(lrl2.predict(Xtrain), Ytrain))
#    l2test.append(accuracy_score(lrl2.predict(Xtest), Ytest))
#graph = [l1, l2, l1test, l2test]
#graph = [l1, l1test]
#color = ["green",  "lightgreen"]
#label = ["L1",  "L1test"]
#plt.figure(figsize=(6, 6))
#for i in range(len(graph)):
#    plt.plot(np.linspace(0.05, 1, 19), graph[i], color[i], label=label[i])
#plt.legend(loc=4) # 图例的位置在哪里?4表示，右下角
#plt.show()

