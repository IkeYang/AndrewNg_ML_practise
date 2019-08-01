#Author:ike yang

##  正规方程得到的结果[[ 4.16333634e-17]
# [ 8.84765988e-01]
# [-5.31788197e-02]]

from scipy.optimize import minimize, rosen, rosen_der
import numpy as np
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
data_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex1\data\ex1data2.txt'

b=np.loadtxt(data_path,delimiter=',')

x_0=b[:,0:2]
y_0=b[:,2]

shape_x=x_0.shape
x_norm=normize(x_0)
y_norm,meany,stdy=normize(y_0)

x_fit=np.ones([shape_x[0],2+1])
x_fit[:,1:3]=x_norm
y_fit=np.zeros([shape_x[0],1])
y_fit[:,0]=y_norm
theta0=np.array([0,0,0])##初始参数
# y_fit.flatten()可以使用这个函数将结果变为一维数组
def lossJ(theta):
    '''

    :param theta: 代入theta 计算loss
    :return: 返回一个实数
    '''
    # theta[1]=-theta[1]
    def h(theta, X):  # Linear hypothesis function
        a=np.dot(X, theta.reshape((3,-1)))
        return a
    x=x_fit
    y=y_fit
    m=x.shape[0]
    v=(h(theta, x) - y).T
    b=(h(theta, x) - y)
    a=np.dot(v,b)
    loss=float(1. / (2 * m)) *a
    return loss
def gradient(theta):
    '''

    :param theta: d在loss函数中对theta求导 拿到数据可以先reshape 再操作 ，对于结果进行flatten
    :return: 返回一个一维数组  注意一定是一维数组
    '''
    m=x_fit.shape[0]
    a=np.dot(x_fit,theta).reshape((-1,1))
    b=np.dot((a-y_fit).T,x_fit)
    c = np.sum(b, axis=0)
    delt=1/m*c*0.01
    return delt

res=minimize(lossJ,theta0,jac=gradient,method='CG')
print(res.x)














