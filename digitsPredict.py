# -*- coding: utf-8 -*-
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
#导入数据
digits = load_digits()
X, y = digits.data, digits.target
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
#建模
gnb = GaussianNB().fit(Xtrain,Ytrain)
#查看分数,calculate the accuracy score
acc_score = gnb.score(Xtest,Ytest)
print(acc_score)
#查看预测结果
Y_pred = gnb.predict(Xtest)
print(Y_pred)
#查看预测的概率结果
prob = gnb.predict_proba(Xtest)
print(prob)
print(prob.shape)
#每一列对应一个标签下的概率
print(prob[1,:].sum())
#每一行的和都是一
print(prob.sum(axis=1))
#使用混淆矩阵来查看贝叶斯结果
from sklearn.metrics import confusion_matrix as CM
print(CM(Ytest, Y_pred))