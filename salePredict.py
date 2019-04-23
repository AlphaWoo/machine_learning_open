# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import GridSearchCV

path = 'D:\Machine Learning\8 practice_Project\Advertising.csv'
data = pd.read_csv(path)
x = data[['TV','Radio','Newspaper']]
y = data['Sales']
print(x)
print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)

for i in [x_test,y_test]:
    i.index = range(i.shape[0])
    
#model = Ridge()
model = Lasso()
alpha_can = np.logspace(-3,2,10)
lasso_model = GridSearchCV(model,param_grid={'alpha':alpha_can},cv=5)
lasso_model.fit(x,y)
print('验证参数：\n',lasso_model.best_params_)

y_hat = lasso_model.predict(np.array(x_test))

mse = np.average((y_hat - np.array(y_test))**2)
rmse = np.sqrt(mse)
print(mse,rmse)
t = np.arange(len(x_test))

lasso_model_=lasso_model.score(x_test,y_test)
plt.plot(t,y_test,'r-',linewidth=2,label='Test')
plt.plot(t,y_hat,'g-',linewidth=2,label='Predict')
plt.legend(loc='upper right')
plt.grid()
plt.show()