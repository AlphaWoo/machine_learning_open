# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = load_breast_cancer() #导入乳腺癌数据集
X = data.data
y = data.target
lrl1 = LR(penalty="l1",solver = "liblinear", C=0.5,max_iter=1000)
lrl2 = LR(penalty="l2",solver = "liblinear", C=0.5,max_iter=1000)