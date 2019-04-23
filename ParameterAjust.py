# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = load_breast_cancer()
rfc = RandomForestClassifier(n_estimators=40,random_state=90)
score_pre = cross_val_score(rfc,data.data,data.target,cv=10).mean()
