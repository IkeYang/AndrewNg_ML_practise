#Author:ike yang
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex2\data\ex2data2.txt'
data = pd.read_csv(data1_path, header=None, names=['x1', 'x2', 'y'])
def data_handle(data):



    m=data['y'].shape[0]
    data['x0']=np.ones([m,1])
    sns.scatterplot(x='x1',y='x2',data=data,hue='y')
    data=data[['x0','x1','x2','y']]
    theta0=np.array([0,0,0])

    degrees=6
    count=1
    for i in range(1, degrees + 1):
        for j in range(0, i + 1):
            term1 = data['x1'].values** (i - j)
            term2 = data['x2'].values ** (j)
            term = (term1 * term2).reshape(term1.shape[0], 1)
            data['x%d'%(count)]=term
            count+=1

    theta0=np.zeros([1,data.shape[1]-1]).flatten()
    names=data.columns.values.tolist()
    names.remove('y')
    # print(names)
    x=data[names]
    y=data['y']
    return x,y,theta0

def sigmoid(x):
    g=1/(1+np.exp(-x))
    return g
def hx(theta,x):
    theta.reshape((-1,1))
    y=np.dot(x,theta)
    return sigmoid(y)
def loss(theta,x,y,l=0):
    # x = data[['x0', 'x1', 'x2']]
    # y = data['y']
    m=y.shape[0]
    x=x.values
    y=y.values.reshape((-1,1))
    theta=theta.reshape((-1,1))
    a=np.log(hx(theta,x))
    left=y*a
    right=(1-y)*np.log(1-hx(theta,x))
    j=-1/m*np.sum(left+right,axis=0)+np.dot(theta[1:].T,theta[1:])*l/2/m
    print(j)
    return j
def gradient(theta,x,y,l):
    # x = data[['x0', 'x1', 'x2']]
    # y = data['y']
    m=y.shape[0]
    x=x.values
    y=y.values.reshape((-1,1))
    theta = theta.reshape((-1, 1))
    mid=np.dot((hx(theta,x)-y).T,x)
    delt=1/m*mid+theta.T*l/m
    return delt.flatten()
def mapFeature( x1col, x2col ):
    """
    Function that takes in a column of n- x1's, a column of n- x2s, and builds
    a n- x 28-dim matrix of featuers as described in the homework assignment
    """
    degrees = 6
    out = np.ones( (x1col.shape[0], 1) )

    for i in range(1, degrees+1):
        for j in range(0, i+1):
            term1 = x1col ** (i-j)
            term2 = x2col ** (j)
            term  = (term1 * term2).reshape( term1.shape[0], 1 )
            out   = np.hstack(( out, term ))
    return out
def plotBoundary(theta, mylambda=0.):
    """
    Function to plot the decision boundary for arbitrary theta, X, y, lambda value
    Inside of this function is feature mapping, and the minimization routine.
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the hypothesis classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """

    xvals = np.linspace(-1,1.5,50)
    yvals = np.linspace(-1,1.5,50)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            myfeaturesij = mapFeature(np.array([xvals[i]]),np.array([yvals[j]]))
            zvals[i][j] = np.dot(theta,myfeaturesij.T)
    zvals = zvals.transpose()#z.T

    u, v = np.meshgrid( xvals, yvals )
    mycontour = plt.contour( xvals, yvals, zvals, [0])
    #Kind of a hacky way to display a text on top of the decision boundary
    myfmt = { 0:'Lambda = %d'%mylambda}
    plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
    plt.title("Decision Boundary")
x,y,theta0=data_handle(data)
# print(y)
# print(x)
# print(loss(theta0,x,y))
res=minimize(loss,theta0,method='L-BFGS-B',args=(x,y,1),options={"maxiter":500, "disp":False} )
print(res.fun)
theta=res.x.reshape((-1,1))
Z=hx(theta,x.values)

Z[Z<0.5]=0
Z[Z>0.5]=1
# print(Z.flatten())
# plt.plot(Z.flatten(),color='red')
# plt.plot(y.values.flatten(),color='blue')

plotBoundary(theta.T,0.)


plt.show()




# h = 0.02
# X=x.values
#
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
# x0=np.ones([xx.shape[0],xx.shape[1]])
# # Z = np.dot(np.c_[xx.ravel(), yy.ravel()], theta) + b
# Z =hx(theta,np.c_[x0.ravel(),xx.ravel(), yy.ravel()])
# # Z =hx(theta,np.c_[x0,xx.ravel(), yy.ravel()])
# # Z = np.argmax(Z)
# Z[Z<0.5]=0
# Z[Z>0.5]=1

# Z = Z.reshape(xx.shape)
#
#
# fig = plt.figure()
# plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)  ##z这种有分类边界的图就是用地形图画的
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.show()















