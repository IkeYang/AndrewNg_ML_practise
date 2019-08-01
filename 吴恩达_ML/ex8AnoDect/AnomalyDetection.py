#Author:ike yang
import matplotlib.pyplot as plt
import numpy as np
import scipy.io #Used to load the OCTAVE *.mat files
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
# data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex8\data\ex8data1.mat'
# data1= scipy.io.loadmat(data1_path)
# data1X=data1['X']
# data1Xval=data1['Xval']
# data1Yval=data1['yval']
data2_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex8\data\ex8data2.mat'
data2= scipy.io.loadmat(data2_path)
print(data2.keys())
data2X=data2['X']
data2Xval=data2['Xval']
data2Yval=data2['yval']
# plt.plot(data1X[:,0],data1X[:,1],'bo')
# plt.scatter(data1X[:,0],data1X[:,1], color='', marker='o', edgecolors='g', s=200)
# plt.show()

class GaussDect():
    def __init__(self,dataX):
        self.dataX=dataX
        N=StandardScaler()
        N.fit(self.dataX)
        self.dataXN = N.transform(self.dataX)
        self.NormP = N
    def mutlFit(self):
        xigema=np.dot(self.dataXN.T, self.dataXN)

    def singelFit(self):
        self.u=np.mean(self.dataXN,axis=0)
        self.var=np.var(self.dataXN,axis=0)
        P=np.ones([self.dataXN.shape[0],])
        for i in range(self.dataXN.shape[1]):
            pI=1/np.sqrt(2*np.pi*self.var[i])*np.exp(-np.square(self.dataXN[:,i]-self.u[i])/self.var[i]).flatten()
            P=P*pI

        self.P=P
        # self.u = np.mean(self.dataX, axis=0)
        # self.var = np.var(self.dataX, axis=0)
        # P = np.ones([self.dataX.shape[0], ])
        # for i in range(self.dataX.shape[1]):
        #     pI = 1 / np.sqrt(2 * np.pi * self.var[i]) * np.exp(
        #         -np.square(self.dataX[:, i] - self.u[i]) / self.var[i]).flatten()
        #     P = P * pI
        #
        # self.P = P
    def chooseE(self,xval,yval,step=0.0001,maxliter=2000):
        P = np.ones([xval.shape[0], ])
        yP = np.zeros([xval.shape[0], ])
        xval=self.NormP.transform(xval)
        for i in range(xval.shape[1]):
            pI = 1 / np.sqrt(2 * np.pi * self.var[i]) * np.exp(
                -np.square(xval[:, i] - self.u[i]) / self.var[i]).flatten()
            P = P * pI
        F1=[]
        E=[]
        for e in range(1,maxliter):
            e=e*step
            yP[P<e]=1
            E.append(e)
            p = precision_score(yval, yP, average='binary')
            r = recall_score(yval, yP, average='binary')
            F1.append(f1_score(yval, yP, average='binary'))
        indexMax=np.argmax(F1)
        self.e=E[indexMax]
    def predit(self):
        yOut=np.zeros([self.dataXN.shape[0], ])
        yOut[self.P<self.e]=1
        return yOut
#ex1


# AD=GaussDect(data1X)
# AD.singelFit()
# AD.chooseE(data1Xval,data1Yval)
# yOut=AD.predit()
# #draw the contour plot
# x = np.linspace(-6, 6, 1000)
# y = np.linspace(-6, 6, 1000)
# X, Y = np.meshgrid(x, y)
# def fun(x,y,u,var):
#     pI = 1 / np.sqrt(2 * np.pi * var[0]) * np.exp(
#         -np.square(x - u[0]) / var[0])*1 / np.sqrt(2 * np.pi * var[1]) * np.exp(
#         -np.square(y - u[1]) / var[1])
#
#     return pI
# #hist to show whether the date is under the Gauss distribution
# # plt.hist(AD.dataX[:,0],5)
# # plt.show()
# # plt.hist(AD.dataX[:,1],5)
# # plt.show()
# ##
# pos = np.array([AD.dataXN[i,:] for i in range(data1X.shape[0]) if yOut[i] == 1])
# neg = np.array([AD.dataXN[i,:] for i in range(data1X.shape[0]) if yOut[i] == 0])
# plt.contour(X, Y, fun(X,Y,AD.u,AD.var), 20, colors = 'black')
# plt.plot(AD.dataXN[:,0],AD.dataXN[:,1],'bo')
# plt.scatter(pos[:,0],pos[:,1], color='', marker='o', edgecolors='g', s=200)
#
# plt.show()

#ex2

AD=GaussDect(data2X)
AD.singelFit()
AD.chooseE(data2Xval,data2Yval,step=10**(-12))
yOut=AD.predit()

print(np.sum(yOut==1))




































