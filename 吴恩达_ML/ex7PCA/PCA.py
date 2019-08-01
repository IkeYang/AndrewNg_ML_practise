#Author:ike yang
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
import random #To pick random images to display
from scipy.special import expit #Vectorized sigmoid function
import pickle
from scipy.optimize import minimize
# data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex7\data\ex7data1.mat'
# # data2_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex7\data\bird_small.png'
# data1= scipy.io.loadmat(data1_path)
# # data2= scipy.misc.imread(data2_path)/256
# x=data1['X']
# plt.plot(x[:,0],x[:,1],'bo')
# # plt.show()
# # np.linalg.svd()


class PCA():
    def __init__(self,dataX,rate=0.99,number=None):
        self.data=dataX
        self.rate=rate
        self.number=number
    def normalize(self):
        '''

        :return: the self.dataN after normalize and its parameter in self.NormP is stranderdScaler struction
        '''
        s=StandardScaler()
        s.fit(self.data)
        self.dataN=s.transform(self.data)
        self.NormP=s
    def calcCov(self):
        return np.dot(self.dataN.T,self.dataN)

    def chooseUreduce(self,U,S):
        if self.number is not None:
            return U[:,0:self.number]
        else:
            sumS=np.sum(S)
            rate=S[0]/sumS
            i=1
            while rate<self.rate:
                rate+=S[i]/sumS
                i+=1
            return U[:,0:i]
    def train(self):
        self.normalize()
        cov=self.calcCov()
        U,S,V=np.linalg.svd(cov)
        Ureduce=self.chooseUreduce(U,S)
        self.Ureduce=Ureduce
    def fit(self,data=None):
        if data is None:
            Z = np.dot( self.dataN,self.Ureduce)
            return Z
        else:
            dataN=self.NormP.transform(data)
            Z = np.dot(dataN, self.Ureduce.T)
            return Z

    def reconstruct(self,Z):
        X=np.dot(Z,self.Ureduce.T)
        return self.NormP.inverse_transform(X)

# myP=PCA(x,number=1)
# myP.train()
# Z=myP.fit()
# Xnex=myP.reconstruct(Z)
# plt.plot(Xnex[:,0],Xnex[:,1],'ro')
# plt.show()
































































