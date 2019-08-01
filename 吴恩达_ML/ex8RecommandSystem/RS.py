#Author:ike yang
import matplotlib.pyplot as plt
import numpy as np
import scipy.io #Used to load the OCTAVE *.mat files
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.optimize import minimize
name='movie_ids.txt'

dataM_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex8\data\ex8_movies.mat'
dataM= scipy.io.loadmat(dataM_path)
dataMP_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex8\data\ex8_movieParams.mat'
dataMP= scipy.io.loadmat(dataMP_path)
#Load Data
R=dataM['R']==1 # transform this martix into boole value
Y=dataM['Y']
num_users=dataMP['num_users']
num_features=dataMP['num_features']
Theta=dataMP['Theta']
X=dataMP['X']
num_movies=dataMP['num_movies']


def flattenParams(myX, myTheta):
    """
    Hand this function an X matrix and a Theta matrix and it will flatten
    it into into one long (nm*nf + nu*nf,1) shaped numpy array
    """
    return np.concatenate((myX.flatten(), myTheta.flatten()))


# A utility function to re-shape the X and Theta will probably come in handy
def reshapeParams(flattened_XandTheta, mynu,mynm, mynf):

    reX = flattened_XandTheta[:int(mynm * mynf)].reshape((mynm, mynf))
    reTheta = flattened_XandTheta[int(mynm * mynf):].reshape((mynu, mynf))

    return reX, reTheta
class recommenderSystem():
    def __init__(self,R,Y,Theta,X):
        self.R=R
        self.Y=Y
        self.Theta=Theta
        self.X=X



    def costJAndGradient(self,Lamuda=0.1):
        J=0
        Gx=np.zeros(self.X.shape)
        GT=np.zeros(self.Theta.shape)
        for i in range(self.Theta.shape[0]):
            theta=self.Theta[i,:].reshape((1,-1))
            x=self.X[self.R[:,i],:]
            yP=np.dot(x,theta.T)
            yO=self.Y[self.R[:,i],i].reshape((-1,1))
            dy=(yP-yO)
            Gx[self.R[:,i],:]=Gx[self.R[:,i],:]+np.dot(dy,theta)
            GT[i,:]=np.sum(np.dot(dy.T,x),axis=0).flatten()+GT[i,:]
            J+=np.sum(dy**2)
        Gx=Gx+self.X*Lamuda
        GT=GT+self.Theta*Lamuda
        myP=flattenParams(Gx,GT)
        return J/2+Lamuda/2*(np.sum(self.Theta**2)+np.sum(self.X**2)),myP

def costJAndGradient(XAndTheta,Y,R,nu,nm,nf,mylambda):
    J=0
    X,Theta=reshapeParams(XAndTheta,nu,nm,nf)
    Gx=np.zeros(X.shape)
    GT=np.zeros(Theta.shape)
    for i in range(Theta.shape[0]):
        theta=Theta[i,:].reshape((1,-1))
        x=X[R[:,i],:]
        yP=np.dot(x,theta.T)
        yO=Y[R[:,i],i].reshape((-1,1))
        dy=(yP-yO)
        Gx[R[:,i],:]=Gx[R[:,i],:]+np.dot(dy,theta)
        GT[i,:]=np.sum(np.dot(dy.T,x),axis=0).flatten()+GT[i,:]
        J+=np.sum(dy**2)
    Gx=Gx+X*mylambda
    GT=GT+Theta*mylambda
    myP=flattenParams(Gx,GT)
    return J/2+mylambda/2*(np.sum(Theta**2)+np.sum(X**2)),myP
my_ratings = np.zeros((1682,1))
my_ratings[0]   = 4
my_ratings[97]  = 2
my_ratings[6]   = 3
my_ratings[11]  = 5
my_ratings[53]  = 4
my_ratings[63]  = 5
my_ratings[65]  = 3
my_ratings[68]  = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5
nf = 10
myR_row = my_ratings > 0
Y = np.hstack((Y,my_ratings))
R = np.hstack((R,myR_row))
nm, nu = Y.shape
X = np.random.rand(nm,nf)
Theta = np.random.rand(nu,nf)
myflat = flattenParams(X, Theta)
mylambda = 10.

result=minimize(costJAndGradient,myflat,jac=True,method='CG',args=(Y,R,nu,nm,nf,mylambda),options={'maxiter':50})
print(result)
resX, resTheta = reshapeParams(result['x'], nm, nu, nf)
prediction_matrix = resX.dot(resTheta.T)
my_predictions = prediction_matrix[:,-1]
data_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex8\data\movie_ids2.txt'
nameList=[]
with open(data_path,encoding='utf-8') as f:
    for line in f:
        nameList.append(line.strip())
index=np.argsort(my_predictions)[::-1][:10]
for i in index:
    print(nameList[int(i)])
# np.argpartition


# RS=recommenderSystem(R,Y,Theta,X)
# print(RS.costJAndGradient(1))
# minimize(lossJ,theta0,jac=gradient,method='CG')





























