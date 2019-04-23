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

from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss
import numpy as np
class_1 = 500
class_2 = 500 #两个类别分别设定500个样本
centers = [[0.0, 0.0], [2.0, 2.0]] #设定两个类别的中心
clusters_std = [0.5, 0.5] #设定两个类别的方差

#生成数据集
X, y = make_blobs(n_samples=[class_1, class_2],centers=centers,
cluster_std=clusters_std,
random_state=0, shuffle=False)
#分割数据集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
#先归一化，保证输入多项式朴素贝叶斯的特征矩阵中不带有负数（次数，频率等都为正数）
mms = MinMaxScaler().fit(Xtrain)
Xtrain_ = mms.transform(Xtrain)
Xtest_ = mms.transform(Xtest)
#建立一个多项式朴素贝叶斯分类器
mnb = MultinomialNB().fit(Xtrain_, Ytrain)
#重要属性：调用根据数据获取的，每个标签类的对数先验概率log(P(Y))
#由于概率永远是在[0,1]之间，因此对数先验概率返回的永远是负值
print("对数先验概率mnb.class_log_prior_=",mnb.class_log_prior_)
print("mnb.class_log_prior_.shape=",mnb.class_log_prior_.shape)
#可以使用np.exp来查看真正的概率值
print("使用np.exp来查看真正的概率值 np.exp(mnb.class_log_prior_)=",np.exp(mnb.class_log_prior_))
# print("np.unique(Ytrain)=",np.unique(Ytrain)) #找出 Ytrain中不同的值
# print("Ytrain.shape = ",Ytrain.shape)
print("根据实际数目计算概率(Ytrain == 0).sum()/Ytrain.shape[0]=",(Ytrain == 0).sum()/Ytrain.shape[0])
print("根据实际数目计算概率 (Ytrain == 1).sum()/Ytrain.shape[0]=",(Ytrain == 1).sum()/Ytrain.shape[0])

#3) 伯努利朴素贝叶斯
from sklearn.naive_bayes import BernoulliNB
#普通来说我们应该使用二值化的类sklearn.preprocessing.Binarizer来将特征一个个二值化 然而这样效率过低，因此我们选择归一化之后直接设置一个阈值
mms = MinMaxScaler().fit(Xtrain)
Xtrain_ = mms.transform(Xtrain)
Xtest_ = mms.transform(Xtest)
#A)不设置二值化
bnl_ = BernoulliNB().fit(Xtrain_, Ytrain) #模型拟合
print("准确分数： bnl.score(Xtest_,Ytest) = ", bnl_.score(Xtest_,Ytest))
#B)设置二值化阈值为0.5
bnl = BernoulliNB(binarize=0.5).fit(Xtrain_, Ytrain) #模型拟合
print("准确分数： bnl.score(Xtest_,Ytest) = ", bnl.score(Xtest_,Ytest))


