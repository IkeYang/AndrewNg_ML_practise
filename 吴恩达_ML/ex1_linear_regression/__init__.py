#Author:ike yang
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data1_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex1\data\ex1data1.txt'
data2_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex1\data\ex1data2.txt'
b=np.loadtxt(data1_path,delimiter=',')
x_0=b[:,0]
y_0=b[:,1]
plt.scatter(x_0,y_0)
plt.show()



