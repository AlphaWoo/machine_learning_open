from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime
data = load_breast_cancer()
X = data.data

y = data.target
import pandas as pd
data = pd.DataFrame(X)
print("统一量纲前", data)
# print(data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)
#统一量刚单位
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
data = pd.DataFrame(X)
print("统一量纲后", data)
"""
        Generate descriptive statistics that summarize the central tendency,
        dispersion and shape of a dataset's distribution, excluding
        ``NaN`` values.
"""
data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
Kernel = ["linear","poly","rbf","sigmoid"]
for kernel in Kernel:
    time0 = time()
    clf= SVC(kernel = kernel
    , gamma="auto"
    , degree = 1
    , cache_size=5000
    ).fit(Xtrain,Ytrain)
    print("The accuracy under kernel %s is %f" % (kernel,clf.score(Xtest,Ytest)))
    print("used time: ",time()-time0)

#1、探索最佳参数
score = []
gamma_range = np.logspace(-10, 1, 50) #返回在对数刻度上均匀间隔的数字
for i in gamma_range:
    clf = SVC(kernel="rbf",gamma = i,cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))
print("max(score), gamma_range[score.index(max(score))] =",max(score), gamma_range[score.index(max(score))])
plt.plot(gamma_range,score)
plt.show()

#2、探索最佳参数，网格搜索
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
time0 = time()
gamma_range = np.logspace(-10,1,20)
coef0_range = np.linspace(0,5,10)
param_grid = dict(gamma = gamma_range,coef0 = coef0_range)

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=420)
#网格搜索
grid = GridSearchCV(SVC(kernel = "poly",degree=1,cache_size=5000),
param_grid=param_grid, cv=cv)
grid.fit(X, y)
print("The best parameters are %s with a score of %0.5f" % (grid.best_params_,grid.best_score_))
print("used time: ", time() - time0)
