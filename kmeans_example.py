import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

def dist_euc(X, Y):
    n, d = np.shape(X)[0], np.shape(X)[1]
    m, d = np.shape(Y)[0], np.shape(Y)[1]
    # for i in range(n):
    #     for j in range(m):
    #         d = np.sqrt(np.sum((X[i] - Y[j])**2))
    diff = np.tile(X[:,np.newaxis,:], (1,m,1)) - np.tile(Y[np.newaxis,:,:], (n,1,1))
    dist = np.sqrt(np.sum(np.square(diff), axis=-1, keepdims=False))
    return dist

# import matplotlib.pyplot as plt


pkl_file = open('data1.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
print(data)
plt.scatter(data[:, 0], data[:, 1], color='r', s=2)
plt.show()

k = 3
num = np.shape(data)[0]
Iteration = 20
# centers = (np.random.rand(k,2)-0.5)*2*10
idx = [int(np.random.rand(1)*num) for i in range(k)]
centers = data[idx]
labels = np.zeros(shape=(num), dtype=int)
colors = ['r', 'g', 'b', 'k', 'c', 'm']
np.random.seed(int(time.time())%1000)
plt.figure()

for it in range(Iteration):
    # calculate distance
    dist = dist_euc(data, centers)
    # update labels
    labels = np.argmin(dist, axis=-1)
    # update centers
    for i in range(k):
        idx = [j for j in range(num) if labels[j] == i]    #每个数据对应的分组不再变化。
        centers[i] = np.mean(data[idx], axis=0)
        # plot
        plt.scatter(data[idx, 0], data[idx, 1], color=colors[i], s=2)
        plt.scatter(centers[i, 0], centers[i, 1], color=colors[i], s=100, marker ='x')
    plt.title('iter = %d'%it)
    plt.show()
#
#
