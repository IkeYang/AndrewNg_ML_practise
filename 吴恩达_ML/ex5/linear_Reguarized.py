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
data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex5\data\ex5data1.mat'
# data2_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex5\data\token.mat'
data1= scipy.io.loadmat(data1_path)
# token= scipy.io.loadmat(data2_path)
# print(data1.keys())
# print(data1.items())
# # print(token)

#x need bias
xTrain=np.insert(data1['X'],0,1,axis=1)
yTrain=data1['y']
xVal=np.insert(data1['Xval'],0,1,axis=1)
yVal=data1['yval']
xTest=np.insert(data1['Xtest'],0,1,axis=1)
yTest=data1['ytest']
class linearRegression():
    def __init__(self,x,y,degree,Lamda=1):
        self.Lamda=Lamda
        self.dataX=x
        self.dataY=y
        self.degree=degree
        # self.xDimension=dimension
        self.m=x.shape[0]
        for i in range(self.degree-1):
            self.dataX=np.insert(self.dataX,self.dataX.shape[1],self.dataX[:,1]*self.dataX[:,-1],axis=1)
    def costFunAndGradient(self,theta):
        # if self.xDimension==1:
        #     self.dataX=self.dataX.reshape((-1,1))
        theta=theta.reshape((-1,1))
        xOut=np.dot(self.dataX,theta)
        cost=xOut-self.dataY
        J=1/2/self.m*np.sum(cost*cost)+self.Lamda/2/self.m*np.sum(theta[1:,:])
        gradient0=np.dot(self.dataX.T,cost)/self.m
        thetaZero=np.copy(theta)[0,:]=0
        gradientOut=gradient0+self.Lamda/self.m*thetaZero
        return J,gradientOut.flatten()
    def train(self,theta0):
        result=minimize(self.costFunAndGradient,x0=theta0,method='CG',jac=True) #必须是一维数组
        self.theta=result['x'].reshape((-1,1))
        self.trainErr=result['fun']
        print(result['fun'])
    def fit(self,x,y=None):
        yOut=np.dot(x,self.theta)
        if y is not None:
            cost=yOut-y
            J = 1 / 2 / y.shape[0] * np.sum(cost * cost) + self.Lamda / 2 / self.m * np.sum(self.theta[1:, :])
            return yOut,J
        return yOut

#show the data and linear regression
degree=2
Lamda=10
theta0=np.ones([degree+1,])


def makePlotX(degree,X=None,down=-60,up=40,step=1):
    if X is None:
        x = np.arange(down, up, step)
        x = np.insert(x.reshape((-1, 1)), 0, 1, axis=1)
        for i in range(degree - 1):
            x= np.insert(x, x.shape[1], x[:, 1] *x[:, -1], axis=1)
        return x
    else:
        for i in range(degree - 1):
            X = np.insert(X, X.shape[1], X[:, 1] * X[:, -1], axis=1)
        return X

# LMy = linearRegression(xTrain, yTrain, degree, Lamda)
# # print(LMy.costFunAndGradient(np.array([[1],[1]])))
# LMy.train(theta0)
# x=makePlotX(degree)
# yOut=LMy.fit(x)
# plt.plot(x[:,1],yOut,color='b')
# plt.plot(xTrain[:,1],yTrain,'bo',color='r')
# plt.show()

##learinig curve
def learningCurve(Lamda):
    trainErr=np.zeros([11,])
    valErr=np.zeros([11,])
    sampleNum=np.zeros([11,])
    for i in range(2,13):
        xTrainEx = xTrain[0:i, :]
        yTrainEx = yTrain[0:i, :]
        LMy = linearRegression(xTrainEx, yTrainEx , degree, Lamda)
        LMy.train(theta0)
        trainErr[i-2]=LMy.trainErr
        x=makePlotX(degree,xVal)
        yOut,valErr[i-2]=LMy.fit(x,yVal)
        sampleNum[i-2]=i
    plt.plot(sampleNum, trainErr,color='b',label='train')
    plt.plot(sampleNum,valErr,color='r',label='CV')
    plt.legend(loc="best")
    plt.show()

#learningrate with lamda
Lamda= [0,0.001,0.003,0.01, 0.03 ,0.1 ,0.3 ,1,3,10]
trainErr=np.zeros([10,])
valErr=np.zeros([10,])
sampleNum=np.zeros([10,])
i=0
for j in Lamda:
    # xTrainEx = xTrain[0:i, :]
    # yTrainEx = yTrain[0:i, :]
    LMy = linearRegression(xTrain, yTrain , degree, j)
    LMy.train(theta0)
    trainErr[i]=LMy.trainErr
    x=makePlotX(degree,xVal)
    yOut,valErr[i]=LMy.fit(x,yVal)
    sampleNum[i]=j
    i+=1
plt.plot(sampleNum, trainErr,color='b',label='train')
plt.plot(sampleNum,valErr,color='r',label='CV')
plt.legend(loc="best")
plt.show()



