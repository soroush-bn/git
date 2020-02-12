import math
import numpy as np
import matplotlib.pyplot as plt
#   print(dir(math))
#help(math.pow)
v=np.array([1,2,3]) # VECTOR
a=np.array([[1,2,3],[4,5,6],[7,8,9],[12,32,54]]) # ye araye 2d (matrix) ba 4 satr va 3 sotoon
#b=np.array()
c=np.arange(1,100,2)
print(c)

e=np.linspace(1,100,30) # too  baze 1 ta 100 behem 30 ta adad ba daseleye mosavi bede
print(e)
d=np.ones((3,4))
print(d)
d3=3*np.ones((3,5),dtype=int)
print(d3)

f=np.random.rand(2,3) #ye matris 2dar 3 ke azash ba adade tasadofi 0TA1 por shode
print(f)
f2=np.random.randn(2,3) # ye matris ba tozie normal
print(f2)
f3=np.random.randn(100,1) # mikhaym bebinim normal mishe ya na
plt.hist(f3,bins=10)
plt.show()

g=np.eye(5) #matris i 5tayi
data=np.genfromtxt(r'C:\Users\Soroush\Desktop\hprice.txt',delimiter=',')

x=data[:,0] #hamash sootoone 0
y=data[:,1]
np.savetxt(r'C:\Users\Soroush\Desktop\xha.txt',x,fmt="%.2f")
np.savetxt(r'C:\Users\Soroush\Desktop\yha.txt',y,fmt="%.2f")