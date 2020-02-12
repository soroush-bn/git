import numpy as np
from numpy.linalg import  pinv #pseudo inverse
np.set_printoptions(precision=2,sign=' ',suppress=True)
import matplotlib.pyplot as plt
import math

#import theano
#import keras
'''
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt

#print(keras.__version__)
y=np.arange(-10,10,.5)
x=np.arange(-10,10,.5)

print(x.shape)
x,y=np.meshgrid(x,y)
x.shape
print(x)
z=x**2+y**2 #tabe hazine ke mikhaym minimize konim

#rasme shekle z
fig =  plt.figure(figsize=(12,8))

ax=fig.gca(projection='3d')
surf=ax.plot_surface(x,y,z,cmap=plt.cm.rainbow)
cset=ax.contourf(x,y,z,zdir='z',offset=0,cmap=plt.cm.rainbow)
fig.colorbar(surf,shrink=.5,aspect=5)
plt.show()'''

plt.style.use('ggplot')

#
# x=np.array([[1,2104,5,1,45],[1,1416,3,2,40],[1,1534,3,2,30],[1,852,2,1,36]])
# print(x)
# y=np.array([[460],[232],[315],[178]])
# print(y)
# #solving
#
# theta=pinv(x.transpose()@x)@x.transpose()@y
# print(theta)

def lowess(x, y, f=2. / 3., iter=3):
    """Robust locally weighted regression.

    Inputs:
       - x, y: dataset
       - f: smoothing parameter
       - iter: number of robustifying iterations
    """
    n = len(x)
    r = int(math.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    y_pred = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = np.linalg.solve(A, b)
            y_pred[i] = beta[0] + beta[1] * x[i]

        residuals = y - y_pred
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return y_pred


# create 100 random sample (x, y)
n = 100
x = np.linspace(0, 2 * math.pi, n)
y = np.sin(x) + 0.3 * np.random.randn(n)



#y_pred = lowess(x, y, f=.35, iter=3)




data = np.genfromtxt(r'C:\Users\Soroush\Desktop\hprice.txt', delimiter=',')
print(data)
X = data[:, 0]  # column 0 from data matrix
print(X)
y = data[:, 1]  # column 1 from data matrix
print(y.shape)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=30, c='r', marker='x')
plt.xlabel('size (Sq. feet)')
plt.ylabel('price (x100 $)')
plt.title('House Dataset')
plt.show()