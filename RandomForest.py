# -*- coding: utf-8 -*-
from sklearn.linear_model import Ridge,LinearRegression,Lasso
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.datasets import fetch_california_housing as fch
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

wine = load_wine()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)
clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
clf = clf.fit(Xtrain,Ytrain)
rfc = rfc.fit(Xtrain,Ytrain)
score_c = clf.score(Xtest,Ytest)
score_r = rfc.score(Xtest,Ytest)
print("Single Tree:{}".format(score_c),"Random Forest:{}".format(score_r))

#dot_data = tree.export_graphviz(clf,feature_names = feature_name,class_names=["琴酒","雪梨","贝尔摩德"]，filled=True,rounded=True)
#graph = graphviz.Source(dot_data)
rfc = RandomForestClassifier(n_estimators=25)
rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10)
clf = DecisionTreeClassifier()
clf_s = cross_val_score(clf,wine.data,wine.target,cv=10)
plt.plot(range(1,11),rfc_s,label = "RandomForest")
plt.plot(range(1,11),clf_s,label = "Decision Tree")
plt.legend()
plt.show()

from scipy.special import comb
np.array([comb(25,i)*(0.2**i)*((1-0.2)**(25-i)) for i in range(13,26)]).sum()

#rfc_l = []
#clf_l = []
#rfc = RandomForestClassifier(n_estimators=25)
#rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10)
#for i in range(10):
#    rfc = RandomForestClassifier(n_estimators=25)
#    rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
#    rfc_l.append(rfc_s)
#    clf = DecisionTreeClassifier()
#    clf_s = cross_val_score(clf,wine.data,wine.target,cv=10).mean()
#    clf_l.append(clf_s)
#    
#plt.plot(range(1,11),rfc_l,label = "Random Forest")
#plt.plot(range(1,11),clf_l,label = "Decision Tree")
#plt.legend()
#plt.show()

rfc = RandomForestClassifier(n_estimators=20,random_state=2)
rfc = rfc.fit(Xtrain, Ytrain) #训练

