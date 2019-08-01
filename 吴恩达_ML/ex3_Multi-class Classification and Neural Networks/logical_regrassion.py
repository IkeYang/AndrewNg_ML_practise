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
x=mat['X']
y=mat["y"]

print(x.shape)
print(y.shape,np.unique(y))
#给x加入一列1
x = np.insert(x,0,1,axis=1)
print(x.shape)

def getDatumImg(row):
    """
    Function that is handed a single np array with shape 1x400,
    crates an image object from it, and returns it
    """
    width, height = 20, 20
    square = row[1:].reshape(width,height)
    return square.T


def displayData(indices_to_display=None):
    """
    Function that picks 100 random rows from X, creates a 20x20 image from each,
    then stitches them together into a 10x10 grid of images, and shows it.
    """
    width, height = 20, 20
    nrows, ncols = 10, 10  #取十行十列的数
    if not indices_to_display:
        indices_to_display = random.sample(range(x.shape[0]), nrows * ncols)

    big_picture = np.zeros([height * nrows, width * ncols])
    y_list=[]
    irow, icol = 0, 0
    for idx in indices_to_display:
        if icol == ncols:
            irow += 1
            icol = 0
        y_list.append(y[idx])
        iimg = getDatumImg(x[idx])
        big_picture[irow * height:irow * height + iimg.shape[0], icol * width:icol * width + iimg.shape[1]] = iimg
        icol += 1
    fig = plt.figure(figsize=(6,6))
    img = scipy.misc.toimage( big_picture ) #将numpyarray 转化成图片
    plt.imshow(img,cmap = cm.Greys_r)
    plt.show()
    print(y_list)
#
# displayData()
def sigmoid(x):
    g=1/(1+np.exp(-x))
    return g
def hx(theta,x):
    theta.reshape((-1,1))
    y=np.dot(x,theta)
    return sigmoid(y)

def loss(theta,x,y,l=0):

    m=y.shape[0]
    y=y.reshape((-1,1))
    theta=theta.reshape((-1,1))
    a = np.log(hx(theta, x))
    left = y * a
    right = (1 - y) * np.log(1 - hx(theta, x))
    j = -1 / m * np.sum(left + right, axis=0) + np.dot(theta[1:].T, theta[1:]) * l / 2 / m
    # print(j)
    return j

def gradient(theta,x,y,l):
    # x = data[['x0', 'x1', 'x2']]
    # y = data['y']
    m=y.shape[0]
    y=y.reshape((-1,1))
    theta = theta.reshape((-1, 1))
    mid=np.dot((hx(theta,x)-y).T,x)
    delt=1/m*mid+theta.T*l/m
    return delt.flatten()

def optmize(x,y,l=0):

    initial_theta = np.zeros((x.shape[1], 1)).reshape(-1)
    Theta = np.zeros((10, x.shape[1]))
    for i in range(10):
        iclass = i if i else 10
        Y=np.array([1 if j==iclass else 0 for j in y])
        res = minimize(loss, initial_theta, method='L-BFGS-B', args=(x, Y, 1), options={"maxiter": 50, "disp": False})
        print(res.x,res.fun)
        Theta[i, :] = res.x
    with open('result', 'wb') as f:
        pickle.dump(Theta, f)


optmize(x,y)















