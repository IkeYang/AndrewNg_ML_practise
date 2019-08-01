#Author:ike yang
import numpy as np
a=np.eye(5)
b= a==1
c=np.array([[1,2,3,4,5],[22,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
#
# print(b.shape)
# print(b)
# print(c[b[:,1],0])
print(c)
print(np.sum(c,axis=0))


data_path=r'F:\python\py书籍\Coursera吴恩达机器学习Matlab和Python代码\Coursera-ML-using-matlab-python-master\coursera_ml_ipynb\ex8\data\movie_ids2.txt'
nameList=[]
with open(data_path,encoding='utf-8') as f:
    for line in f:
        nameList.append(line.strip())

print(nameList[np.argmax(my_predictions)])