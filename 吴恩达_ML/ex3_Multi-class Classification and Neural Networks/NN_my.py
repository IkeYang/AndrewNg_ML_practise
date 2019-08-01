#Author:ike yang
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
def sigmoid(x):
    g=1/(1+np.exp(-x))
    return g
def hx(theta,x):
    theta.reshape((-1,1))
    y=np.dot(x,theta)
    return sigmoid(y)
data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex3\data\ex3data1.mat'
mat = scipy.io.loadmat(data1_path)
x=mat['X']
y=mat["y"]
data1_path2=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex3\data\ex3weights.mat'
wimat = scipy.io.loadmat(data1_path2)
print(x.shape)
print(y.shape,np.unique(y))
#给x加入一列1
x = np.insert(x,0,1,axis=1)
print(x.shape)
# print(wimat.keys())
theta1=wimat['Theta1']
theta2=wimat["Theta2"]
print(theta1.shape,theta2.shape)

##不包括偏执b
input_layer_size = 400  #401
hidden_layer_size = 25  #26
output_layer_size = 10
n_training_samples =x.shape[0]  #5000

#x(5000, 401)
#(5000, 1) y 对y进行编号化  使之成为（5000,10)
def code_y(y):
    y_code=np.zeros([y.shape[0],len(np.unique(y))])
    a=y.flatten()-1
    j=0
    for i in a :
        y_code[j,i]=1
        j+=1
    return  y_code

y_code=code_y(y)
#初始化权重
theta10=np.random.rand(hidden_layer_size,input_layer_size+1)#(25*401)
theta20=np.random.rand(output_layer_size,hidden_layer_size+1) #(10*26)
x=np.array(x.flatten()).reshape((-1,1))

def reshapeX(flattenedX):
    return np.array(flattenedX).reshape((n_training_samples,input_layer_size+1))


def ForwaordPrapropagate(theta1,theta2,xrow):
    para_record=[]
    z1=np.dot(theta1,xrow)
    a1=expit(z1)
    para_record.append((z1,a1))
    a1 = np.insert(a1, 0, 1)
    z2=np.dot(theta2,a1)
    a2=expit(z2)
    para_record.append((z2,a2))
    return para_record

def costJ(theta,x,y,l):
    theta1=theta[0:(input_layer_size+1)*hidden_layer_size].reshape(hidden_layer_size,input_layer_size+1)
    theta2=theta[(input_layer_size+1)*hidden_layer_size:].reshape(output_layer_size,hidden_layer_size+1)
    x_my=reshapeX(x)
    total_cost = 0.
    m = n_training_samples
    for row in range(m):
        ixrow=x_my[row,:]
        yreal=y[row,:]
        ypred=ForwaordPrapropagate(theta1,theta2,ixrow)[-1][1]
        # print(ypred.shape)
        mycost = -yreal.T.dot(np.log(ypred))-(1-yreal.T).dot(np.log(1-ypred))
        total_cost += mycost
    total_cost = float(total_cost) / m


    total_reg = 0.
    total_reg += np.sum(theta1 * theta1)
    total_reg += np.sum(theta2 * theta2)
    total_reg *= float(l) / (2 * m)

    return total_cost + total_reg

theta   ·=np.concatenate((theta1.flatten(),theta2.flatten())).flatten()
print(theta.shape)
# print(theta[0:(input_layer_size+1)*hidden_layer_size+1].reshape(hidden_layer_size,input_layer_size+1))
print(costJ(theta,x,y_code,1))



















































