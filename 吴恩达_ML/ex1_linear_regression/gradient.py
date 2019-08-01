#Author:ike yang
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def gradient_theta(a,theta,x,y):
    m=x.shape[0]
    b=(np.dot(x,theta)-y)*x
    delt=-a*1/m*np.sum(b,axis=0)
    c=delt.reshape((-1,1))
    return theta+c
def normize(x):
    if x.ndim==1:
        mean=np.mean(x)
        std=np.std(x)
        a=(x-mean)/std
        return a,mean,std
    else:
        l=x.shape[1]
        a=np.zeros([x.shape[0],x.shape[1]])
        for i in range(l):
            b=x[:,i]
            mean = np.mean(b)
            std = np.std(b)
            a[:,i] = (b - mean) / std
        return a
def h(theta, X):  # Linear hypothesis function
    return np.dot(X, theta)

def lossJ(theta,x,y):
    m=x.shape[0]
    v=(h(theta, x) - y).T
    b=(h(theta, x) - y)
    a=np.dot(v, b)

    return float(1. / (2 * m)) *a










