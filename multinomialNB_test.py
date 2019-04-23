# -*- coding: utf-8 -*-
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
'''
scikit中的make_blobs方法常被用来生成聚类算法的测试数据，直观地说，
make_blobs会根据用户指定的特征数量、中心点数量、范围等来生成几类数据，这些数据可用于测试聚类算
法的效果。
n_samples是待生成的样本的总数。
n_features是每个样本的特征数,默认2。
centers表示类别数。
cluster_std表示每个类别的方差，
例如我们希望生成2类数据，其中一类比另一类具有更大的方差，可以将cluster_std设置为[1.0,3.0]。
'''
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