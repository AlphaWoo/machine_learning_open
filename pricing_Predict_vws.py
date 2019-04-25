# -*- coding: utf-8 -*-
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
from pandas import DataFrame,Series
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime
import sklearn

#data = load_boston()
datafile = u'D:\\Machine Learning\\11 Homework_20190420\\Real_estate_valuation_data_set.xlsx'#文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
data = pd.read_excel(datafile)#datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv

examDf = DataFrame(data)
examDf.head()
exam_X = examDf.iloc[: , 1:-1].values
exam_Y = examDf.iloc[:,-1].values
Xtrain,Xtest,Ytrain,Ytest = TTS(exam_X,exam_Y,train_size=.3)#X_train为训练数据标签,X_test为测试数据标签,exam_X为样本特征,exam_y为样本标签，train_size 训练数据占比

#X = data.data
#y = data.target
#Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)
reg = XGBR(n_estimators=100)

#CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()
##来查看一下sklearn中所有的模型评估指标

sorted(sklearn.metrics.SCORERS.keys())
#使用随机森林和线性回归进行一个对比
rfr = RFR(n_estimators=100)
CVS(rfr,Xtrain,Ytrain,cv=5).mean()
#rfr_score = CVS(rfr,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()
rfr_score = CVS(rfr,Xtrain,Ytrain,cv=5,scoring='mean_squared_error').mean()
lr = LinearR()
CVS(lr,Xtrain,Ytrain,cv=5).mean()
#lr_score = CVS(lr,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()
lr_score = CVS(lr,Xtrain,Ytrain,cv=5,scoring='mean_squared_error').mean()
#开启参数slient：在数据巨大，预料到算法运行会非常缓慢的时候可以使用这个参数来监控模型的训练进度
reg = XGBR(n_estimators=10,silent=False)
xgbr_score = CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()
#xgbr_score = CVS(reg,Xtrain,Ytrain,cv=5).mean()