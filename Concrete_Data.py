# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
#from sklearn.linear_model import LinearRegression as LR
#from sklearn.logistic_model import LogisticRegression as LR
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#读取文件
datafile = u'D:\Machine Learning\9 Homework\Concrete_Data.xls'#文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
data = pd.read_excel(datafile)#datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
examDf = DataFrame(data)
examDf.head()
exam_X = examDf.iloc[: , :-1].values
exam_Y = examDf.iloc[:,-1].values
X_train,X_test,Y_train,Y_test = TTS(exam_X,exam_Y,train_size=.3)#X_train为训练数据标签,X_test为测试数据标签,exam_X为样本特征,exam_y为样本标签，train_size 训练数据占比




#clf = DecisionTreeRegressor(random_state=0)
#rfr = RandomForestRegressor(n_estimators=50,random_state=0)
#clf = clf.fit(X_train,Y_train)
#rfr = rfr.fit(X_train,Y_train)
#score_c = clf.score(X_test,Y_test)
#score_r = rfr.score(X_test,Y_test)

from sklearn.model_selection import cross_val_score
scorel = []
for i in range(10,100):
    rfr = RandomForestRegressor(n_estimators=i,n_jobs=-1,random_state=430)
#    clf = clf.fit(X_train,Y_train)
    rfr = rfr.fit(X_train,Y_train)
#    score_c = clf.score(X_test,Y_test)
    score = rfr.score(X_test,Y_test)
    
#    score = cross_val_score(rfr,data.data,data.target,cv=10).mean()
    scorel.append(score)
print(max(scorel),([*range(10,100)][scorel.index(max(scorel))]))
plt.figure(figsize=[20,5])
plt.plot(range(10,100),scorel)
plt.show()



#
##for i in [X_train, X_test]:
##    i.index = range(i.shape[0])
##Xtrain.shape
#reg = LR().fit(X_train, Y_train)
#yhat = reg.predict(X_test)
##print (yhat)
#from sklearn.metrics import mean_squared_error as MSE
#mse = MSE(yhat,Y_test)
##exam_Y.max()
##exam_Y.min()
#import sklearn
##sorted(sklearn.metrics.SCORERS.keys())
#cvs = CVS(reg,exam_X,exam_Y,cv=10,scoring="neg_mean_squared_error")
#
#from sklearn.metrics import r2_score
#score_ = r2_score(yhat,Y_test)
#r2 = reg.score(X_test,Y_test)
##r3 = reg.score(yhat,Y_test)
#
#
#
##r1=RLR()
##r1.fit(x,y)
##r1.get_support(indices=True)
##print(dataf.columns[r1.get_support(indices=True)])
##t=dataf[dataf.columns[r1.get_support(indices=True)]].as_matrix()
##r2=LR()
##r2.fit(t,y)
##print("训练结束")
##
##print("模型正确率:"+str(r2.score(t,y)))
#
## -*- coding: utf-8 -*-
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from pandas import DataFrame,Series
##from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
#from sklearn.linear_model import LinearRegression as LR
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import load_wine
#from sklearn.impute import SimpleImputer
##from sklearn.datasets import load_breast_cancer_wisconsin
#wine = load_wine()
#
##读取文件
##datafile = u'D:/Machine Learning/9 Homework/breast-cancer-wisconsin.cvs'#文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
#data = pd.read_csv('D:\\Machine Learning\\9 Homework\\breast-cancer-wisconsin-2.csv') # 正样本数据
#
#data = data.replace(to_replace = "?", value = np.nan)
## then drop the missing value
#data = data.dropna(how = 'any')
#
#exam_X = data.iloc[:, :-1].values
#exam_Y = data.iloc[:,-1].values
#Xtrain, Xtest, Ytrain, Ytest = TTS(exam_X,exam_Y,test_size=0.3)
#clf = DecisionTreeClassifier(random_state=0)
#rfc = RandomForestClassifier(random_state=0)
#clf = clf.fit(Xtrain,Ytrain)
#rfc = rfc.fit(Xtrain,Ytrain)
#score_c = clf.score(Xtest,Ytest)
#score_r = rfc.score(Xtest,Ytest)
