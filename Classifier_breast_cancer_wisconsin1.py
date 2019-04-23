# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#import tensorflow as tf
column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size',
                'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',
               'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data = pd.read_csv('breast-cancer-train.csv',names=column_names)
data.head()
data.info()
data = data.replace(to_replace='?',value=np.nan)
data = data.dropna(how='any')
data.shape

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

print(X_train.shape,Y_train.shape)

lr = LogisticRegression()
lr.fit(x_train,y_train)
y_predict = lr.predict(x_test)

from sklearn.metrics import classification_report
print(lr.score(x_test,y_test))