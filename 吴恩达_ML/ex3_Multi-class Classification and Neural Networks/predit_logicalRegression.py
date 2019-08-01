#Author:ike yang
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
import random #To pick random images to display
from scipy.special import expit #Vectorized sigmoid function
import pickle
from scipy.optimize import minimize
data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex3\data\ex3data1.mat'
mat = scipy.io.loadmat(data1_path)
x=mat['X']
y=mat["y"]

print(x.shape)
print(y.shape,np.unique(y))
#给x加入一列1
x = np.insert(x,0,1,axis=1)
print(x.shape)
with open('result', 'rb') as f:
    Theta = pickle.load(f)
def sigmoid(x):
    g=1/(1+np.exp(-x))
    return g
def hx(theta,x):
    theta.reshape((-1,1))
    y=np.dot(x,theta)
    return sigmoid(y)
def predit_mylogicalRegression(theta,x):
    lenth=10
    score=np.zeros([x.shape[0],lenth])
    for i in range(lenth):
        score[:,i]=hx(theta[i,:],x)
    max_idence=np.argmax(score,axis=1)
    max_idence=np.array([j if j!=0 else 10 for j in max_idence])
    return max_idence
    print(max_idence,np.unique(max_idence))

y_pred=predit_mylogicalRegression(Theta,x)
def socre(y_pred,y_real):
    y_pred=y_pred.flatten()
    y_real=y_real.flatten()
    l=len(y_real)
    a=np.sum(y_real==y_pred)
    print(l,a)
    return a/l
result=socre(y_pred,y)
print(result)






