# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression as LR

#path = 'D:\Machine Learning\8 practice_Project\Advertising.csv'
#data = pd.read_csv(path)
#x = data[['TV','Radio','Newspaper']]
#y = data['Sales']
#
#plt.plot(data['TV'],y,'ro',label='TV')
#plt.plot(data['Radio'],y,'g^',label='Radio')
#plt.plot(data['Newspaper'],y,'mv',label='Newspaper')
#plt.legend(loc='lower right')
#plt.grid()
#plt.show()


path = 'D:\Machine Learning\8 practice_Project\Advertising.csv'
Advertising = pd.read_csv(path)
#housevalue = fch() #会需要下载，大家可以提前运行试试看
#X = pd.DataFrame(Advertising.Sales) #放入DataFrame中便于查看
#y = Advertising.target
X = pd.DataFrame(Advertising.Sales) #放入DataFrame中便于查看
y = Advertising.target
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])
Xtrain.shape
reg = LR().fit(Xtrain, Ytrain)
yhat = reg.predict(Xtest)

from sklearn.metrics import mean_squared_error as MSE
MSE(yhat,Ytest)
y.max()
y.min()
import sklearn
sorted(sklearn.metrics.SCORERS.keys())
cross_val_score(reg,X,y,cv=10,scoring="neg_mean_squared_error")
from sklearn.metrics import r2_score
r2_score(yhat,Ytest)
r2 = reg.score(Xtest,Ytest)
r2
r2_score(Ytest,yhat)
r2_score(y_true = Ytest,y_pred = yhat)
cross_val_score(reg,X,y,cv=10,scoring="r2").mean()
import matplotlib.pyplot as plt
sorted(Ytest)
plt.plot(range(len(Ytest)),sorted(Ytest),c="black",label= "Data")
plt.plot(range(len(yhat)),sorted(yhat),c="red",label = "Predict")
plt.legend()
plt.show()

#
#x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
#model = LR()
#model = model.fit(x_train,y_train)
#ypred=model.predict(x_test)
##mse = np.average((yhat - np.array(y_test))**2)
##rmse = np.sqrt(mse)
##print(mse,rmse)
#
#train_score = model.score(x_train,y_train)
#test_score = model.score(x_test,y_test)
#print('train score:{train_score:.6f};test score:{test_score:.6f}'.format(train_score=train_score,test_score=test_score))
#
#
#t = np.arange(len(x_test))
#
#plt.plot(range(len(y_test)),y_test,c="black",label="Data")
#plt.plot(range(len(ypred)),ypred,c="red",label="Predict")
#model_ = LR.score(x_test,y)
#plt.legend()
#plt.show()






# -*- coding: utf-8 -*-
#from sklearn.linear_model import LinearRegression as LR
#from sklearn.model_selection import train_test_split

#from sklearn.datasets import fetch_california_housing as fch 
#import pandas as pd
#housevalue = fch() #会需要下载，大家可以提前运行试试看
#X = pd.DataFrame(housevalue.data) #放入DataFrame中便于查看
#y = housevalue.target
#X.shape
#y.shape
#X.head()
#housevalue.feature_names
#X.columns = housevalue.feature_names
#"""
#MedInc：该街区住户的收入中位数
#HouseAge：该街区房屋使用年代的中位数
#"""
#"""
#AveRooms：该街区平均的房间数目
#AveBedrms：该街区平均的卧室数目
#Population：街区人口
#AveOccup：平均入住率
#Latitude：街区的纬度
#Longitude：街区的经度
#"""
#Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
#for i in [Xtrain, Xtest]:
#    i.index = range(i.shape[0])
#Xtrain.shape
#reg = LR().fit(Xtrain, Ytrain)
#yhat = reg.predict(Xtest)
#yhat
#reg.coef_
#[*zip(Xtrain.columns,reg.coef_)]
#reg.intercept_
#from sklearn.metrics import mean_squared_error as MSE
#MSE(yhat,Ytest)
#y.max()
#y.min()
#import sklearn
#sorted(sklearn.metrics.SCORERS.keys())
#cross_val_score(reg,X,y,cv=10,scoring="neg_mean_squared_error")
#from sklearn.metrics import r2_score
#r2_score(yhat,Ytest)
#r2 = reg.score(Xtest,Ytest)
#r2
#r2_score(Ytest,yhat)
#r2_score(y_true = Ytest,y_pred = yhat)
#cross_val_score(reg,X,y,cv=10,scoring="r2").mean()
#import matplotlib.pyplot as plt
#sorted(Ytest)
#plt.plot(range(len(Ytest)),sorted(Ytest),c="black",label= "Data")
#plt.plot(range(len(yhat)),sorted(yhat),c="red",label = "Predict")
#plt.legend()
#plt.show()