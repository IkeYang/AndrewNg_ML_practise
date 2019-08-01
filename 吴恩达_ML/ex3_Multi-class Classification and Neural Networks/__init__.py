#Author:ike yang
import numpy as np
from scipy.optimize import minimize
# x=np.array([[1,2,3],[1,2,3],[1,2,3]])
z=np.array([[1,2,3]]).reshape((-1,1))
h=np.array([[2,3,5]]).reshape((-1,1))
print(1/h)
# print(np.argmin(h,axis=0))
# print(type(None) is None)
# # print(np.sum(np.square(h-z)))
# # print(h*z)
# print(h)
# z=np.zeros([h.shape[0],h.shape[1]])
# # z=0
print(z,h)
# pri、nt(z)
# h = np.insert(h,0,z[:,0],axis=1)
# print(h.T)
# print('the%dLayer'%2)
# print(x.shape)
# print(x)
# # print(np.array([1]).shape)
# y=x.flatten()
# print(y.shape)
# print(y.reshape((3,-1)))
# print(np.reshape(y,(3,3)).shape)
#
class funM():
    def __init__(self):
        pass
    def Min(self,x,a,b,c):
        return a * x * x + b * x + c,np.array([2 * x * a + b])

    def GMin(self,x, a, b, c):
        return np.array([2 * x * a + b])
    def MinMize(self):
        return minimize(self.Min,x0=np.array([2]),args=(1,2,1),method='CG',jac=self.GMin)
def Min(x,a,b,c):
    return a*x*x+b*x+c
def GMin(x,a,b,c):
    return np.array([2*x*a+b])
# a=funM()
# print(minimize(a.Min,x0=np.array([2]),args=(1,2,1),method='CG',jac=True)['x'])

# print(a.MinMize())
# print(np.ones([22,]).shape)
# print(np.dot(x,x))
# print(np.random.random([3,]).shape)