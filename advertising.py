# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

path = 'D:\Machine Learning\8 practice_Project\Advertising.csv'
data = pd.read_csv(path)
x = data[['TV','Radio','Newspaper']]
y = data['Sales']

plt.plot(data['TV'],y,'ro',label='TV')
plt.plot(data['Radio'],y,'g^',label='Radio')
plt.plot(data['Newspaper'],y,'mv',label='Newspaper')
plt.legend(loc='lower right')
plt.grid()
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
linreg = LinearRegression()
model = linreg.fit(x_train,y_train)
yhat=model.predict(x_test)
mse = np.average((yhat - np.array(y_test))**2)
rmse = np.sqrt(mse)
print(mse,rmse)

t = np.arange(len(x_test))

plt.plot(range(len(y_test)),y_test,c="black",label="Data")
plt.plot(range(len(yhat)),yhat,c="red",label="Predict")
model_ = linreg.score(x_test,y_test)
plt.legend()
plt.show()