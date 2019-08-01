#Author:ike
import numpy as np
import matplotlib.pyplot as plt
from gradient import *


data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex1\data\ex1data1.txt'
data2_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex1\data\ex1data2.txt'
b=np.loadtxt(data1_path,delimiter=',')
b=b.reshape((-1,2))
x_0=b[:,0]
x_0,x2,x1=normize(x_0)

shape_x=x_0.shape
y_0=b[:,1]
# y_0,my,stdy=normize(y_0)
y_fit=np.zeros([shape_x[0],1])
y_fit[:,0]=y_0
x_fit=np.ones([shape_x[0],1+1])
x_fit[:,1]=x_0[:]
jvec=[]
theta=np.zeros([2,1])
a=0.01
err=0.0001
err2=0.000000001
literation=1000
j = lossJ(theta, x_fit, y_fit)
jvec.append(j)
for i in range(literation):
    theta = gradient_theta(a, theta, x_fit, y_fit)
    j=lossJ(theta,x_fit,y_fit)
    print(j)
    jvec.append(j)
    if j<err or abs(jvec[i-1]-jvec[i])<err2:
        print(i)
        break
print(theta)






x_0=b[:,0]
# x_fit[:,1]=x_0[:]
y_pred=np.dot(x_fit,theta)
y_0=b[:,1]
plt.scatter(x_0,y_0)
plt.plot(x_0,y_pred)
plt.show()