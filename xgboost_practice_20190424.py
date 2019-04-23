# -*- coding: utf-8 -*-
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score,mean_squared_error as MSE
##默认reg:linear
#reg = XGBR(n_estimators=180,random_state=420).fit(Xtrain,Ytrain)
#reg.score(Xtest,Ytest)
#MSE(Ytest,reg.predict(Xtest))
##xgb实现法
#import xgboost as xgb
##使用类Dmatrix读取数据
#dtrain = xgb.DMatrix(Xtrain,Ytrain)
#dtest = xgb.DMatrix(Xtest,Ytest)
##非常遗憾无法打开来查看，所以通常都是先读到pandas里面查看之后再放到DMatrix中
#dtrain
##写明参数，silent默认为False，通常需要手动将它关闭
#param = {'silent':False,'objective':'reg:linear',"eta":0.1}
#num_round = 180
##类train，可以直接导入的参数是训练数据，树的数量，其他参数都需要通过params来导入
#bst = xgb.train(param, dtrain, num_round)
##接口predict

data = load_boston()
#波士顿数据集非常简单，但它所涉及到的问题却很多
X = data.data
y = data.target
Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)
#r2_score(Ytest,bst.predict(dtest))
#MSE(Ytest,bst.predict(dtest))
cv=5
reg = XGBR(n_estimators=180,random_state=420).fit(Xtrain,Ytrain)
param = {"reg_alpha":np.arange(0,5,0.05),"reg_lambda":np.arange(0,2,0.05)}
gscv = GridSearchCV(reg,param_grid = param,scoring = "neg_mean_squared_error",cv=cv)
#======【TIME WARNING：10~20 mins】======#
time0=time()
gscv.fit(Xtrain,Ytrain)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
gscv.best_params_
gscv.best_score_
preds = gscv.predict(Xtest)
r2_score(Ytest,preds)
MSE(Ytest,preds)