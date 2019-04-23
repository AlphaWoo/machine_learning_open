# -*- coding: utf-8 -*-
import pandas as pd
data = pd.read_csv(r"D:\Machine Learning\8 practice_Project\Advertising.csv")
#X = data.iloc[:,1:]
#y = data.iloc[:,0]
#X.shape

from sklearn.feature_selection import VarianceThreshold
import numpy as np
#selector = VarianceThreshold() #实例化，不填参数默认方差为0
#X_var0 = selector.fit_transform(X) #获取删除不合格特征之后的新特征矩阵
##也可以直接写成 X = VairanceThreshold().fit_transform(X)
#X_var0.shape
#

X = data.iloc[:,1:]
y = data.iloc[:,0]
X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
##假设在这里我一直我需要300个特征
X_fschi = SelectKBest(chi2, k=2).fit_transform(X_fsvar, y)
#X_fschi.shape
#
cross_val_score(RFC(n_estimators=10,random_state=0),X_fschi,y,cv=1).mean()