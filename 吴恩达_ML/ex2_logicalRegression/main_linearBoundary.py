#Author:ike yang
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der

data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex2\data\ex2data1.txt'
b=np.loadtxt(data1_path,delimiter=',')
data=pd.read_csv(data1_path,header=None,names=['x1','x2','y'])
m=data['y'].shape[0]
data['x0']=np.ones([m,1])
# x=b[:,0:2]
# y=b[:,2]
sns.scatterplot(x='x1',y='x2',data=data,hue='y')

theta0=np.array([0,0,0])
x=data[['x0','x1','x2']]
y=data['y']
def sigmoid(x):
    g=1/(1+np.exp(-x))
    return g
def hx(theta,x):
    theta.reshape((-1,1))
    y=np.dot(x,theta)
    return sigmoid(y)
def loss(theta):
    x = data[['x0', 'x1', 'x2']]
    y = data['y']
    m=y.shape[0]
    x=x.values
    y=y.values.reshape((-1,1))
    theta=theta.reshape((-1,1))
    a=np.log(hx(theta,x))
    left=y*a
    right=np.multiply(1-y,np.log(1-hx(theta,x)))
    j=-1/m*np.sum(left+right,axis=0)
    return j
def gradient(theta):
    x = data[['x0', 'x1', 'x2']]
    y = data['y']
    m=y.shape[0]
    x=x.values
    y=y.values.reshape((-1,1))
    theta = theta.reshape((-1, 1))
    mid=np.dot((hx(theta,x)-y).T,x)
    delt=1/m*mid
    return delt.flatten()
def cacl_x2(x1,theta):
    o0=theta[0]
    o1=theta[1]
    o2=theta[2]
    x2=-o1/o2*x1-o0/o2
    return x2


res=minimize(loss,theta0,jac=gradient,method='CG')
print(res.x)
minx=data['x1'].min()
maxx=data['x1'].max()
x1=np.linspace(minx,maxx,100)
x2=cacl_x2(x1,theta=res.x)
plt.plot(x1,x2,color='black')

plt.show()











































