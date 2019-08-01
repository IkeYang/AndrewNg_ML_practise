#Author:ike yang
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
import random #To pick random images to display
from scipy.special import expit #Vectorized sigmoid function
import pickle
from scipy.optimize import minimize
data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex7\data\ex7data1.mat'
# data2_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex7\data\bird_small.png'
data1= scipy.io.loadmat(data1_path)
# data2= scipy.misc.imread(data2_path)/256
print(data1)
# np.linalg.svd()