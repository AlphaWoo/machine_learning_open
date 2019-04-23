#1. 导入需要的模块
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

#2. 实例化数据集，可视化数据集
X,y = make_blobs(n_samples=50, centers=2, random_state=0,cluster_std=0.6)
print("X=",X)
print("X[:,0]=",X[:,0])
print("X[:,1]=",X[:,1])

#3. 画决策边界：理解函数contour
# matplotlib.axes.Axes.contour([X, Y,] Z, [levels], **kwargs)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plt.xticks([])
plt.yticks([])
# plt.show()

#首先要有散点图
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
ax = plt.gca() #获取当前的子图，如果不存在，则创建新的子图

#获取平面上两条坐标轴的最大值和最小值
xlim = ax.get_xlim()
ylim = ax.get_ylim()
#在最大值和最小值之间形成30个规律的数据
axisx = np.linspace(xlim[0],xlim[1],30)
axisy = np.linspace(ylim[0],ylim[1],30)
print("axisy=",axisy)
axisy,axisx = np.meshgrid(axisy,axisx)
#我们将使用这里形成的二维数组作为我们contour函数中的X和Y
#使用meshgrid函数将两个一维向量转换为特征矩阵
#核心是将两个特征向量广播(brocasting)，以便获取y.shape * x.shape这么多个坐标点的横坐标和纵坐标
xy = np.vstack([axisx.ravel(), axisy.ravel()]).T
#其中ravel()是降维函数，vstack能够将多个结构一致的一维数组按行堆叠起来
#xy就是已经形成的网格，它是遍布在整个画布上的密集的点
plt.scatter(xy[:,0],xy[:,1],s=1,cmap="rainbow")
#理解函数meshgrid和vstack的作用
a = np.array([1,2,3])
b = np.array([7,8])
#两两组合，会得到多少个坐标？
#答案是6个，分别是 (1,7),(2,7),(3,7),(1,8),(2,8),(3,8)
v1,v2 = np.meshgrid(a,b)
# v1
# v2
v = np.vstack([v1.ravel(), v2.ravel()]).T

#建模，通过fit计算出对应的决策边界
clf = SVC(kernel = "linear").fit(X,y)
Z = clf.decision_function(xy).reshape(axisx.shape)
#重要接口decision_function，返回每个输入的样本所对应的到决策边界的距离
#然后再将这个距离转换为axisx的结构，这是由于画图的函数contour要求Z的结构必须与X和Y保持一致
#画决策边界和平行于决策边界的超平面
ax.contour(axisx,axisy,Z
            ,colors="k"
            ,levels=[-1,0,1] #画三条等高线，分别是Z为-1，Z为0和Z为1的三条线
            ,alpha=0.5
            ,linestyles=["--","-","--"])
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.show()

#记得Z的本质么？是输入的样本到决策边界的距离，而contour函数中的level其实是输入了这个距离
#让我们用一个点来试试看
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plt.scatter(X[10,0],X[10,1],c="black",s=50,cmap="rainbow")
clf.decision_function(X[10].reshape(1,2))
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
ax = plt.gca()
ax.contour(axisx,axisy,Z,colors="k",levels=[-3.33917354] ,alpha=0.5,linestyles=["--"])

# plt.show()

# 探索

#根据决策边界，对X中的样本进行分类，返回的结构为n_samples
print("clf.predict(X)=",clf.predict(X))
#返回给定测试数据和标签的平均准确度
print("clf.score(X,y)=", clf.score(X,y))
#返回支持向量
print("clf.support_vectors_ = ", clf.support_vectors_)
#返回每个类中支持向量的个数
print("clf.n_support_ = ", clf.n_support_)