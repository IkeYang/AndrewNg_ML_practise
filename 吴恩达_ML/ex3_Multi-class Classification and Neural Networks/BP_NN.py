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
x=mat['X'] #(5000, 400)
x = np.insert(x,0,1,axis=1)#(5000，401)
y=mat["y"]#(5000, 1)
data=np.append(y,x,axis=1)
np.random.shuffle(data)
x=data[:,1:]
y=data[:,0]
datafile = r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex3\data\ex3weights.mat'
mat = scipy.io.loadmat( datafile )
Theta1, Theta2 = mat['Theta1'], mat['Theta2']

def changeY(y,num):
    '''

    :param y:
    :param num:num of category
    :return:
    '''
    yAfter=np.zeros([y.shape[0],num])
    for i in range(y.shape[0]):
        yAfter[i,int((y[i]-1)%10)]=1
    return yAfter
y=changeY(y,10)#(500,10)
# zhu
# hiddenLayer=1


class BP_NN():
    def __init__(self,dataX,dataY,hiddenLayer,hiddenLayerNum=None,method='classify'):
        '''

        :param dataX: X
        :param dataY: Y
        :param hiddenLayer: number of hidden Layer
        :param hiddenLayerNum: num of neure in each hidden Layer. If it is None, then set is equals 3*dataX.shape[0]
        '''
        self.dataX=dataX
        self.dataY=dataY
        self.shapeX=dataX.shape
        self.shapeY=dataY.shape
        self.hiddenLayer=hiddenLayer
        self.thetaParam=None
        self.method=method
        if hiddenLayerNum==None:
            self.hiddenLayerNum=3*dataX.shape[1]
        else:
            self.hiddenLayerNum = hiddenLayerNum
    def initTheta(self):
        '''

        :return: the theta0, whose shape is (n,) and in the random set of [0,0.1) and n equals the whole num summed up.
        '''
        theta={}
        for i in range(self.hiddenLayer+1):
            if i==0:
                num=self.shapeX[1]*self.hiddenLayerNum
                theta['0']=np.random.random([num,])*(2*np.sqrt(6)/(np.sqrt(self.dataX.shape[1]+self.hiddenLayerNum)))-(np.sqrt(6)/(np.sqrt(self.dataX.shape[1]+self.hiddenLayerNum)))
                continue
            elif i==self.hiddenLayer:
                num = (self.hiddenLayerNum+1)*self.shapeY[1]
                theta[str(i)] = np.random.random([num, ]) * (
                            2 * np.sqrt(6) / (np.sqrt(self.dataY.shape[1] + self.hiddenLayerNum+1))) - (
                                         np.sqrt(6) / (np.sqrt(self.dataY.shape[1] + self.hiddenLayerNum+1)))
                continue
            else:
                num = (self.hiddenLayerNum + 1) * (self.hiddenLayerNum)
                theta[str(i)] = np.random.random([num, ]) * (
                        2 * np.sqrt(6) / (np.sqrt( self.hiddenLayerNum*2+1))) - (
                                        np.sqrt(6) / (np.sqrt(+ self.hiddenLayerNum*2+1)))
        theta0=self.strechTheta(theta)
        # totalNum=(self.shapeX[1])*self.hiddenLayerNum+(self.hiddenLayerNum+1)*self.hiddenLayerNum*(self.hiddenLayer-1)+(self.hiddenLayerNum+1)*self.shapeY[1]
        # theta0=np.random.random([totalNum,])*(2*np.sqrt(6)/(np.sqrt(self.dataX.shape[1]+self.hiddenLayerNum)))-(np.sqrt(6)/(np.sqrt(self.dataX.shape[1]+self.hiddenLayerNum)))
        return theta0
    # #set the random seeds
    #
    # np.random.seed(1234)
    def GFun(self,x,function='sigment'):
        if function=='sigment':
            return expit(x)
    def DGFun(self,x,function='sigment'):
        if function=='sigment':
            return expit(x)*(1-expit(x))

    def thetaL(self,theta,l):
            '''

            :param l: 0 input -1 output l interHiddenLayer
            :return:matrix
            '''
            if l == 0:  # input->hidden1
                return theta[0:(self.shapeX[1]) * self.hiddenLayerNum].reshape((self.hiddenLayerNum, self.shapeX[1]))
            if l == -1:  # hiddenEnd->outPut
                return theta[-(self.hiddenLayerNum + 1) * self.shapeY[1]:].reshape(
                    (self.shapeY[1], self.hiddenLayerNum + 1))
            return theta[(self.shapeX[1]) * self.hiddenLayerNum + (self.hiddenLayerNum + 1) * self.hiddenLayerNum * (
                        l - 1) + 1:(self.shapeX[1]) * self.hiddenLayerNum + (
                        self.hiddenLayerNum + 1) * self.hiddenLayerNum * (l) + 1].reshape(
                (self.hiddenLayerNum, self.hiddenLayerNum + 1))

    def forwardPropagation(self, theta, dataXSlice,vector=False):
        '''

        :param theta:
        :param dataXSlice: one dimision of X
        :return: a dict that include a and z
        '''
        if vector==False:
            Out={'InputToHidden':None,'HiddenToHidden':None,'HiddenToOutput':None}
            thetaIn = self.thetaL(theta, 0)
            thetaOutput = self.thetaL(theta, -1)
            z1 = np.dot(thetaIn, dataXSlice.reshape((-1, 1)))
            a1 = self.GFun(z1)
            a1 = np.insert(a1, 0, 1, axis=0)
            Out['InputToHidden']={'a':a1,'z':z1}
            if self.hiddenLayer - 1 != 0:  # hidden>1 use this
                aHid = np.zeros([self.hiddenLayerNum, self.hiddenLayer - 1])
                zHid = np.zeros([self.hiddenLayerNum, self.hiddenLayer - 1])
                aHid = np.insert(aHid, 0, 1, axis=0)
                xForm = a1
                for l in range(self.hiddenLayer - 1):
                    thetal = self.thetaL(theta, l + 1)
                    zHid[:, l] = np.dot(thetal, xForm).flatten()
                    aHid[1:, l] = self.GFun(zHid[:, l])
                    xForm = aHid[:, l]
                Out['HiddenToHidden'] = {'a': aHid, 'z': zHid}
            else:
                aHid = a1
                zHid = z1

            Outz = np.dot(thetaOutput, aHid[:, -1].reshape((-1, 1)))
            OutPut = self.GFun(Outz)
            Out['HiddenToOutput'] = {'a': OutPut, 'z': Outz}
            return Out
        else:
            yOut=np.zeros([dataXSlice.shape[0],self.dataY.shape[1]])
            for i in range(dataXSlice.shape[0]):
                thetaIn = self.thetaL(theta, 0)
                thetaOutput = self.thetaL(theta, -1)
                z1 = np.dot(thetaIn, dataXSlice[i,:].reshape((-1, 1)))
                a1 = self.GFun(z1)
                a1 = np.insert(a1, 0, 1, axis=0)
                if self.hiddenLayer - 1 != 0:  # hidden>1 use this
                    aHid = np.zeros([self.hiddenLayerNum, self.hiddenLayer - 1])
                    zHid = np.zeros([self.hiddenLayerNum, self.hiddenLayer - 1])
                    aHid = np.insert(aHid, 0, 1, axis=0)
                    xForm = a1
                    for l in range(self.hiddenLayer - 1):
                        thetal = self.thetaL(theta, l + 1)
                        zHid[:, l] = np.dot(thetal, xForm).flatten()
                        aHid[1:, l] = self.GFun(zHid[:, l])
                        xForm = aHid[:, l]

                else:
                    aHid = a1
                    zHid = z1

                Outz = np.dot(thetaOutput, aHid[:, -1].reshape((-1, 1)))
                OutPut = self.GFun(Outz)
                yOut[i, :] = OutPut.reshape((1, -1))
                if self.method=='classify':
                    index=np.argmax(yOut[i,:])
                    yOut[i,:]=0
                    yOut[i,index]=1

            return yOut
    def prepareAZ(self,Out):
        '''

        :param Out: the struct of forwardProga
        :return: dict A,Z with num 0,1,2,... max index= hidlayer Num
        '''
        z={}
        a={}
        a['1']=Out['InputToHidden']['a']
        z['1']=Out['InputToHidden']['z']
        for i in range(self.hiddenLayer-1):
            a[str(i+2)]=Out['InputToHidden']['a'][:,i].reshape((-1,1))
            z[str(i+2)]=Out['InputToHidden']['z'][:,i].reshape((-1,1))
        # a[str(self.hiddenLayer)] = Out['HiddenToOutput']['a']
        # z[str(self.hiddenLayer)] = Out['HiddenToOutput']['z']
        return a,z
    def strechTheta(self,dTheta):
        '''

        :param dTheta:
        :return:
        '''
        dThetaF=dTheta['0'].flatten()
        for i in range(self.hiddenLayer):
            dThetaF=np.append(dThetaF,dTheta[str(i+1)].flatten())
        return dThetaF

    def backPropagation(self,theta,Out,ySlice,XSlice):
        '''

        :param theta:
        :param Out: the struct calculated by the forward propagation
        :param ySlice: the result of xSlice
        :return:dTheta
        '''
        # diyata={'InputToHidden':None,'HiddenToHidden':None,'HiddenToOutput':None}
        diyata={} #diyata from out layer(hiddenLayer + 1) to input layer and  input layer(0) donot have diyata
        dTheta={}  #dTheta from input layer(0) to output layer and output layer donot have dTheta

        # diyata[str(self.hiddenLayer + 1)]=(ySlice.reshape((-1,1))-Out['HiddenToOutput']['a'].reshape((-1,1)))*self.DGFun(Out['HiddenToOutput']['z'].reshape((-1,1)))
        diyata[str(self.hiddenLayer + 1)]=(ySlice.reshape((-1,1))-Out['HiddenToOutput']['a'].reshape((-1,1)))
        formerD=diyata[str(self.hiddenLayer + 1)]
        a,z=self.prepareAZ(Out)
        a['0']=XSlice.reshape((-1,1))
        dTheta[str(self.hiddenLayer)] =np.dot(diyata[str(self.hiddenLayer+1)],a[str(self.hiddenLayer)].T)
        for i in range(self.hiddenLayer):
            name=str(self.hiddenLayer -i)
            if i==0:
                numO=-1
            else:
                numO=self.hiddenLayer-i
            # DLin = np.dot(self.thetaL(theta, numO)[:,1:.T , formerD )* self.DGFun(z[str(self.hiddenLayer-i)]).reshape((-1, 1))  # input theta
            DLin = np.dot(self.thetaL(theta, numO).T , formerD )* np.insert(self.DGFun(z[str(self.hiddenLayer-i)]).reshape((-1, 1)), 0, 0, axis=0)  # input theta
            DLin=np.delete(DLin,0,axis=0)
            # Dbias = formerD
            diyata[name] = DLin
            formerD=DLin
            dThetaNormal=np.dot(DLin,a[str(self.hiddenLayer-1-i)].T)
            # Dbias = DLin
            # dTheta[str(self.hiddenLayer-i-1)] = np.insert(dThetaNormal, 0, Dbias, axis=1)
            dTheta[str(self.hiddenLayer-i-1)] = dThetaNormal

        dThetaV=self.strechTheta(dTheta)
        return dThetaV
        #
        # if self.hiddenLayer<2:
        #     DLin=self.thetaL(theta,0)[:,1:].T*diyata['HiddenToOutput']*self.DGFun(Out['InputToHidden']['z']).reshape((-1,1))#input theta
        #     Dbias=diyata['HiddenToOutput']
        #     diyata['InputToHidden']=np.insert(DLin,0,Dbias,axis=1)
        #     return diyata
        # else:
    def costFunAndGradient(self,theta,lamda=0.2):
        '''

        :param theta: (n,)
        :return: cost , Gradient
        '''
        #divide theta into on or two
        J=0
        dThetaT=np.zeros([theta.shape[0],])
#行列转换 以及加一行
        for i in range(self.shapeX[0]):#Forwara progagtion
            OneOut=self.forwardPropagation(theta,self.dataX[i,:])
            yk=self.dataY[i,:].reshape((-1,1))
            hox=OneOut['HiddenToOutput']['a'].reshape((-1,1))
            J += np.sum(-yk*np.log(hox)-(1-yk)*np.log(1-hox))
            # J+=0.5*np.sum((yk-hox)*(yk-hox))
            dTheta=self.backPropagation(theta,OneOut,self.dataY[i,:],self.dataX[i,:])
            dThetaT+=dTheta
        J=J/self.dataY.shape[0]+lamda/2/self.dataY.shape[0]*np.sum(theta*theta)
        dTheta=-dThetaT/self.dataY.shape[0]-lamda*dThetaT
        return J, dTheta
    def gradintCheck(self, theta0, disturbance=0.0001):
        result=np.zeros([theta0.shape[0],2])
        for i in range(theta0.shape[0]):
            theta=np.copy(theta0)
            thetaUp=np.copy(theta0)
            thetaUp[i]=thetaUp[i]+disturbance
            thetaLow = np.copy(theta0)
            thetaLow[i] = thetaLow[i] -disturbance
            JUP,G=self.costFunAndGradient(thetaUp,lamda=0)
            JLow,G=self.costFunAndGradient(thetaLow,lamda=0)
            J,G=self.costFunAndGradient(theta0,lamda=0)
            result[i,0]=(JUP-JLow)/(2*disturbance)
            result[i,1]=G[i]
        return result
    def trainNN(self,theta0=None):
        if theta0 is None:
            theta0=self.initTheta()
        # resultG=self.gradintCheck(theta0)
        # result=minimize(self.costFunAndGradient,x0=theta0,method='BFGS',jac=True)
        # self.thetaParam=result['x']
        self.thetaParam=self.Netown(theta0)
    def distanceX(self,x1,x2):
        return np.sum(np.abs(x1-x2))
    def Netown(self,theta0,stop=0.0001,maxIter=200,rate=0.0005):
        JOld,G=self.costFunAndGradient(theta0)
        # print(J)
        times=1
        theta1=theta0-G*rate
        # dis=self.distanceX(theta0, theta1)
        JNew, G = self.costFunAndGradient(theta1)
        theta1 = theta0 - G * rate
        while abs(JOld-JNew)>stop and times<maxIter:
            JOld=np.copy(JNew)
            JNew, G = self.costFunAndGradient(theta1)
            print(JNew)
            times += 1
            theta0=theta1
            theta1 = theta0 -G*rate
            # dis = self.distanceX(theta0, theta1)
        return theta1
    def fitNN(self,x,y):
        '''

        :param x: (num of sample, dim of sample)
        :return:
        '''
        if self.thetaParam.all==None:
            raise NotImplementedError
        Out=self.forwardPropagation(self.thetaParam,x,vector=True)
        rightNum=0
        wrongNum=0
        for i in range(x.shape[0]):
            if np.argmax(y[i,:])==np.argmax(Out[i,:]):
                rightNum+=1
            else:
                wrongNum+=1
        return rightNum/(rightNum+wrongNum)

nn=BP_NN(x,y,1,25)
# theta={}
# theta['0']=Theta1
# theta['1']=Theta2
# theta0=nn.strechTheta(theta)
# nn.trainNN(theta0)
nn.trainNN()
print(nn.fitNN(x,y))
# theta0=nn.strechTheta(theta)
# a,b=nn.costFunAndGradient(theta0,lamda=0)
# nn.trainNN()
# nn.costFun(nn.initTheta())












































































