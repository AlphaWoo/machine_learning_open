# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,LinearRegression,Lasso
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing as fch
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

housevalue=fch()
X=pd.DataFrame(housevalue.data)
Y=housevalue.target

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.3,random_state=420)
alpharange=np.arange(1,201,10)
ridge,lr=[],[]

for alpha in alpharange:
    reg=Ridge(alpha=alpha)
    linear=LinearRegression()
    regs = cross_val_score(reg,X,Y,cv=5,scoring = "r2").mean()
    linears = cross_val_score(linear,X,Y,cv=5,scoring = "r2").mean()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpharange,ridge,color = "red",label = "Ridge")
plt.plot(alpharange,lr,color = "orange",label = "LR")
plt.title("Mean")
plt.legend()
plt.show()

_alphas = np.logspace(-10,-2,200,base = 10)
lasso_ = LassoCV(alphas=_alphas,cv=5).fit(X,Y)
alpha_=lasso_.alpha_

lasso_.mse_path_
lasso_.mse_path_.shape
Msd_=lasso_.mse_path_.mean(axis=1)

coef_ = lasso_.coef_
Score_=lasso_.score(Xtest,Ytest)
