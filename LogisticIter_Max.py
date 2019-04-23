# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer() # 导入乳腺癌数据集
X = data.data
y = data.target
#使用L2正则化
l2 = []
l2test = []
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
for i in np.arange(1,201,10): #start=1, stop=201, step=10
    lrl2 = LR(penalty="l2",solver="liblinear",C=0.9,max_iter=i) #得到当前 max_iter下的 L2正则化的逻辑回归
    lrl2 = lrl2.fit(Xtrain,Ytrain) #拟合数据得到模型
    l2.append(accuracy_score(lrl2.predict(Xtrain),Ytrain)) #存储训练准确率
    l2test.append(accuracy_score(lrl2.predict(Xtest),Ytest)) #存储测试准确率
#画图
graph = [l2,l2test]
color = ["black","gray"]
label = ["L2","L2test"]
plt.figure(figsize=(20,5))
for i in range(len(graph)):
    plt.plot(np.arange(1,201,10),graph[i],color[i],label=label[i])
plt.legend(loc=4) # 图例的位置在哪里?4表示，右下角
plt.xticks(np.arange(1,201,10)) # X轴
plt.show()
#若设max_iter=300，我们可以使用 属性.n_iter_来调用本次求解中真正实现的迭代次数
lr = LR(penalty="l2",solver="liblinear",C=0.9,max_iter=300).fit(Xtrain,Ytrain)
print("lr.n_iter_=", lr.n_iter_)