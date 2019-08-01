#Author:ike yang
#Author:ike yang
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
import random #To pick random images to display
from scipy.special import expit #Vectorized sigmoid function
import pickle
from scipy.optimize import minimize
data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex7\data\ex7data2.mat'
data2_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex7\data\bird_small.png'
data1= scipy.io.loadmat(data1_path)
data2= scipy.misc.imread(data2_path)/256
# print(data2[0,0,:])
# plt.imshow(data2)
# plt.show()
# print(data1)
# x=data1['X']




class Kmeans():
    def __init__(self,K,dataX):
        self.K=K
        self.dataX=dataX
    def initK(self):
        '''

        :return: uK is a dict which name is No. and item is array and it assign class which is a dict has name of No. and is empty
        '''
        uK={}
        Class ={}
        dataXIndex=np.random.randint(self.dataX.shape[0], size=(self.K, 1))
        for i in range(self.K):
            uK[str(i)]=self.dataX[dataXIndex[i,0],:].reshape((1,-1))
            Class[str(i)]=[]
        return uK,Class
    def sortX(self,uK,xSlice):
        '''

        :param uK:
        :param xSlice:
        :return: No, of X's class
        '''
        distance=float('inf')

        for i in range(self.K):
            cost=uK[str(i)].reshape((1,-1))-xSlice.reshape((1,-1))
            disNew=np.dot(cost,cost.T)
            if disNew<distance:
                distance=disNew
                index=i
        return index
    def calcMeanUk(self,uK,Class):
        '''

        :param Class: with each Class's X index
        :return: the mean of X in each class and make in a dict like initK
        '''
        uKnew={}
        for i in range(self.K):
            if Class[str(i)]==[]:
                uKnew[str(i)]=self.dataX[np.random.randint(self.dataX.shape[0]),:].reshape((1,-1))
            else:
                uKnew[str(i)]=np.mean(self.dataX[Class[str(i)],:],axis=0).reshape((1,-1))
        return uKnew

    def calcCost(self,xClass,uK):
        '''

        :param xClass:
        :param uK:
        :return: cost of 1 fanshu
        '''
        J=0
        for i in range(self.dataX.shape[0]):
            cost=np.sum(np.abs(self.dataX[i,:].reshape((1,-1))-uK[str(int(xClass[i,0]))].reshape((1,-1))))
            J+=cost
        return J/self.dataX.shape[0]/2
    def cmpDict(self,d1,d2):
        for i in range(self.K):
            if d1[str(i)].all()!=d2[str(i)].all():
                return False
        return True

    def trainK(self):
        uK,ClassIndex=self.initK()
        uKtrend =np.copy(uK)
        xClass=np.zeros([self.dataX.shape[0],1])
        while True:
            for i in range(self.dataX.shape[0]):
                res=self.sortX(uK,self.dataX[i,:])
                xClass[i, 0]=res
                ClassIndex[str(res)].append(i)

            uKNew=self.calcMeanUk(uK,ClassIndex)
            if  self.cmpDict(uKNew,uK):
                break
            else:
                for i in range(self.K):
                    uKtrend[str(i)]=np.insert(uKtrend[str(i)],uKtrend[str(i)].shape[0],uKNew,axis=0)
                uK=uKNew
        self.uK=uK
        J=self.calcCost(xClass,uK)
        return xClass,J
    def predict(self,x,uK=None):
        if uK is None:
            if x.shape[0]!=1:
                xClass=np.zeros([x.shape[0],1])
                for i in range(x.shape[0]):
                    xClass[i,0]=self.sortX(self.uK,x[i,:])
                return xClass
            xClass = self.sortX(self.uK, x)
            return xClass
        if x.shape[0] != 1:
            xClass = np.zeros([x.shape[0], 1])
            for i in range(x.shape[0]):
                xClass[i, 0] = self.sortX(uK, x[i, :])
            return xClass
        xClass = self.sortX(self.uK, x)
        return xClass
# def plotBoundary(my_model, xmin, xmax, ymin, ymax,uK=None):
#     """
#     Function to plot the decision boundary for a trained SVM
#     It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
#     And for each, computing whether the SVM classifies that point as
#     True or False. Then, a contour is drawn with a built-in pyplot function.
#     """
#     xvals = np.linspace(xmin,xmax,100)
#     yvals = np.linspace(ymin,ymax,100)
#     zvals = np.zeros((len(xvals),len(yvals)))
#     for i in range(len(xvals)):
#         for j in range(len(yvals)):
#             if uK is None:
#                 zvals[i][j] = float(my_model.predict(np.array([xvals[i],yvals[j]]).reshape(((1,-1)))))
#             else:
#                 zvals[i][j] = float(my_model.predict(np.array([xvals[i], yvals[j]]).reshape(((1, -1))),uK))
#     zvals = zvals.transpose()
#
#     u, v = np.meshgrid( xvals, yvals )
#     plt.contour( u, v, zvals)
# Jlist=[]
# uKList=[]
# xClassList=[]
# for i in range(100):
#     # plt.plot(x[:,0],x[:,1],'bo')
#     myK=Kmeans(3,x)
#     xClass,J=myK.trainK()
#     Jlist.append(J)
#     uKList.append(myK.uK)
#     xClassList.append(xClass)
# index=np.argmin(Jlist)
# #     # plotBoundary(myK,-1,10,0,7)
# #     # plt.title("Decision Boundary is "+str(J)+'    '+str(i))
# #     # plt.show()
# # print(np.argmin(Jlist),Jlist[np.argmin(Jlist)])
# # print(uKList[np.argmin(Jlist)])
# # uk=uKList[np.argmin(Jlist)]
# # plt.plot(x[:,0],x[:,1],'bo')
# # pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
# # neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])
# # for i in range(3):
# #     plt.plot(uk[str(i)][0,0],uk[str(i)][0,1],'ro')
# # plotBoundary(myK,-1,10,0,7,uKList[np.argmin(Jlist)])
# # plt.title("Decision Boundary is "+str(Jlist[np.argmin(Jlist)]))
# # plt.show()
# #
# # myK=Kmeans(3,x)
# # xClass,J=myK.trainK()
# # C0 = np.array([x[i,:] for i in range(x.shape[0]) if xClass[i,0] == 0])
# # C1 = np.array([x[i,:] for i in range(x.shape[0]) if xClass[i,0]== 1])
# # C2 = np.array([x[i,:] for i in range(x.shape[0]) if xClass[i,0]== 2])
# # plt.plot(C0[:,0],C0[:,1],'ro',label='0')
# # plt.plot(C1[:,0],C1[:,1],'yo',label='1')
# # plt.plot(C2[:,0],C2[:,1],'bo',label='2')
# # uk=myK.uK
# # for i in range(3):
# #     plt.plot(uk[str(i)][0,0],uk[str(i)][0,1],'b+',MarkerSize='20')
# #
# # plt.legend(loc="best")
# #
# # plt.show()
#
#
#
# print(Jlist[index])
# xClass=xClassList[index]
# C0 = np.array([x[i,:] for i in range(x.shape[0]) if xClass[i,0] == 0])
# C1 = np.array([x[i,:] for i in range(x.shape[0]) if xClass[i,0]== 1])
# C2 = np.array([x[i,:] for i in range(x.shape[0]) if xClass[i,0]== 2])
# plt.plot(C0[:,0],C0[:,1],'ro',label='0')
# plt.plot(C1[:,0],C1[:,1],'yo',label='1')
# plt.plot(C2[:,0],C2[:,1],'mo',label='2')
# uk=uKList[index]
# for i in range(3):
#     plt.plot(uk[str(i)][0,0],uk[str(i)][0,1],'k+',MarkerSize='20')
#
# plt.legend(loc="best")
#
# plt.show()


x=data2.reshape((-1,3))
pict=Kmeans(16,x)
xClass,J=pict.trainK()
uk=pict.uK
# AOut=np.array([uk[str(int(xClass[i,0]))] for i in range(x.shape[0])])
AOut=np.zeros([x.shape[0],3])
for i in range(x.shape[0]):
    AOut[i,:]=uk[str(int(xClass[i,0]))].flatten()
plt.imshow(AOut.reshape(128,128,3))
plt.show()



