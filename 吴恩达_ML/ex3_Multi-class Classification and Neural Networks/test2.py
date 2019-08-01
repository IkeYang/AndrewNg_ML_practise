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
data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex3\data\ex3data1.mat'
mat = scipy.io.loadmat(data1_path)
x=mat['X'] #(5000, 400)
x = np.insert(x,0,1,axis=1)#(5000，401)
y=mat["y"]#(5000, 1)
data=np.append(y,x,axis=1)
np.random.shuffle(data)
x=data[:,1:]
y=data[:,0]
datafile = r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex3\data\ex3weights.mat'
mat = scipy.io.loadmat( datafile )
Theta1, Theta2 = mat['Theta1'], mat['Theta2']
print(theta1)