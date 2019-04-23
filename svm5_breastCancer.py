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

print("X.shape =", X.shape)
print("X = ", X)

np.unique(y)
plt.scatter(X[:,0],X[:,1],c=y)

print("X[:,0] =", X[:,0])
print("X[:,1] =", X[:,1])
# plt.show()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
# Kernel = ["linear","poly","rbf","sigmoid"]
Kernel = ["linear","rbf","sigmoid"]
for kernel in Kernel:
    time0 = time()
    clf= SVC(kernel = kernel, gamma="auto"
        ,degree = 2
        , cache_size=5000
    ).fit(Xtrain,Ytrain)
    print("The accuracy under kernel %s is %f" % (kernel,clf.score(Xtest,Ytest)))
    print("used time: ",time()-time0)


#
# Kernel = ["linear","rbf","sigmoid"]
# for kernel in Kernel:
#     time0 = time()
#     clf= SVC(kernel = kernel
#     , gamma="auto"
#     # , degree = 1
#     , cache_size=5000
#     ).fit(Xtrain,Ytrain)
#     print("The accuracy under kernel %s is %f" % (kernel,clf.score(Xtest,Ytest)))
#     print("used time: ",time()-time0)