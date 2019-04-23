# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.linear_model import LinearRegression as LR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
#from sklearn.datasets import load_breast_cancer_wisconsin
wine = load_wine()

#读取文件
#datafile = u'D:/Machine Learning/9 Homework/breast-cancer-wisconsin.cvs'#文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
data = pd.read_csv('D:\\Machine Learning\\9 Homework\\breast-cancer-wisconsin-2.csv') # 正样本数据

data = data.replace(to_replace = "?", value = np.nan)
# then drop the missing value
data = data.dropna(how = 'any')

exam_X = data.iloc[:, :-1].values
exam_Y = data.iloc[:,-1].values
Xtrain, Xtest, Ytrain, Ytest = TTS(exam_X,exam_Y,test_size=0.3)
clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
clf = clf.fit(Xtrain,Ytrain)
rfc = rfc.fit(Xtrain,Ytrain)
score_c = clf.score(Xtest,Ytest)
score_r = rfc.score(Xtest,Ytest)