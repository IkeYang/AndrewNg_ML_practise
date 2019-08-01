#Author:ike yang
import numpy as np
import gradient
import matplotlib.pyplot as plt
def normalEquation(x,y):
    theta=np.linalg.pinv(x.T@x)@x.T@y
    return theta
data_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex1\data\ex1data2.txt'

b=np.loadtxt(data_path,delimiter=',')

x_0=b[:,0:2]
y_0=b[:,2]

shape_x=x_0.shape
x_norm=gradient.normize(x_0)
y_norm,meany,stdy=gradient.normize(y_0)

x_fit=np.ones([shape_x[0],2+1])
x_fit[:,1:3]=x_norm
y_fit=np.zeros([shape_x[0],1])
y_fit[:,0]=y_norm
jvec=[]
theta=np.zeros([3,1])
a=0.01
err=0.00001
err2=0.0000000001
literation=10000
j = gradient.lossJ(theta, x_fit, y_fit)
print(j)
jvec.append(j)
for i in range(literation):
    theta = gradient.gradient_theta(a, theta, x_fit, y_fit)
    print(theta)
    j=gradient.lossJ(theta,x_fit,y_fit)
    print(j)
    jvec.append(j)
    if j<err or abs(jvec[i-1]-jvec[i])<err2:
        print(i+2)
        break
# print(theta)
jvec=np.array(jvec).reshape((-1,1))
# print(jvec.shape)
plt.plot(jvec)
plt.show()

theta2=normalEquation(x_fit,y_fit)
print(theta,theta2)











