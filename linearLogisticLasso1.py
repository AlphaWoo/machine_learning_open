# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,LinearRegression,Lasso
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.datasets import fetch_california_housing as fch
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

housevalue = fch()
X = pd.DataFrame(housevalue.data)
Y = housevalue.target

Xtrain,Xtest,Ytrain, 