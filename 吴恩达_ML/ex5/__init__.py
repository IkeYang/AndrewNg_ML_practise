#Author:ike yang

import numpy as np
h=np.array([[2,3,5]]).reshape((-1,1))
z=np.array([[1,1,1]]).reshape((-1,1))
# print(h,z)
h=np.insert(h,h.shape[1],z[:,0],axis=1)
print(h)
h=np.insert(h,h.shape[1],z[:,0],axis=1)
print(h)#在第i=1 列处插入z z必须为一个一维
print(np.insert(h,h.shape[1],z[:,0],axis=1))#在第i=1 列处插入z z必须为一个一维
# print(np.insert(h,1,z,axis=0))#在第i=1 行处插入z

print(np.array([[1],[1]]).shape)


















