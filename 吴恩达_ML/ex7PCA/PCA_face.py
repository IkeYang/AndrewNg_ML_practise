#Author:ike yang
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
from PCA import PCA
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale
data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex7\data\ex7faces.mat'
dataFace= scipy.io.loadmat(data1_path)['X']
# print(dataFace.shape)
def imageShow(dataFace,pictN=25):
    for i in range(pictN):
        plt.subplot(np.sqrt(pictN),np.sqrt(pictN),i+1)
        data=dataFace[i,:].reshape((32,32)).T
        # img = scipy.misc.toimage(data)
        plt.imshow(data,cmap='gray')
    plt.show()
imageShow(dataFace)
# myP=PCA(x,number=1)
# myP.train()
# Z=myP.fit()
# Xnex=myP.reconstruct(Z)
# plt.plot(Xnex[:,0],Xnex[:,1],'ro')
# plt.show()
for i in [5,10,50,100,500]:
    myP=PCA(dataFace,number=i)
    myP.train()
    Z=myP.fit()
    Xnex=myP.reconstruct(Z)
    imageShow(Xnex)




















