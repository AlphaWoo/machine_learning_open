# -*- coding: utf-8 -*-
#调整max_depth
param_grid = {'max_depth':np.arange(1, 20, 1)}
# 一般根据数据的大小来进行一个试探，乳腺癌数据很小，所以可以采用1~10，或者1~20这样的试探
# 但对于像digit recognition那样的大型数据来说，我们应该尝试30~50层深度（或许还不足够
#   更应该画出学习曲线，来观察深度对模型的影响
rfc = RandomForestClassifier(n_estimators=39
                             ,random_state=90
                           )
GS = GridSearchCV(rfc,param_grid,cv=10)#cv:cross value times=10
GS.fit(data.data,data.target)
GS.best_params_
GS.best_score_