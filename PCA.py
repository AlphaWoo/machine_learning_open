# -*- coding: utf-8 -*-
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

#path = 'D:\Machine Learning\8 practice_Project\Advertising.csv'
#data = pd.read_csv(path)

def loadDataSet(fileName,delim='\t'):
 fr=open(fileName)
 stringArr=[line.strip().split(delim)for line in fr.readlines()]
 dataArr=[map(float,line )for line in stringArr]
return mat(dataArr)
def pca(dataMat,topNfeat=9999999):
 meanVals=mean(dataMat,axis=0)
 meanRemoved=dataMat-meanVals
 covMat=cov(meanRemoved,rowvar=0)
 eigvals,eigVects=linalg.eig(mat(covMat))
 eigValInd=argsort(eigvals)
 eigValInd=eigValInd[:-(topNfeat+1):-1]
 redEigVects=eigVects[:,eigValInd]
 lowDDataMat=meanRemoved*redEigVects
 reconMat=(lowDDataMat*redEigVects.T)+meanVals
return lowDDataMat,reconMat
def display():
import PCA
dataMat=PCA.loadDataSet('D:\Machine Learning\8 practice_Project\Advertising.csv')
lowDmat,reconMat=PCA.pca(dataMat,1)
print shape(lowDmat)
print dataMat
print reconMat

 fig=plt.figure()
 ax=fig.add_subplot(111)
 ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
 ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50,c='red')
 plt.show()
if __name__=='__main__':
 d=loadDataSet("testSet.txt")
 f=pca(d)
print f 
