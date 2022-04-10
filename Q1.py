import library
from copy import deepcopy
import numpy as np

a,b,c,d,e,f = np.loadtxt("mat1.txt" , unpack = True)
A1=[]
for i in range(len(a)):
    A1.append([a[i],b[i],c[i],d[i],e[i],f[i]])

B1 = [[19],[2],[13],[-7],[-9],[2]]


A1t = deepcopy(A1)
B1t = deepcopy(B1)


library.luDecompose(A1t,len(A1t))
library.forwardBackwardSubstitution(A1t,B1t)
ans11 = deepcopy(B1t)

A1t = deepcopy(A1)
B1t = deepcopy(B1)

# Resetting the matrices
for i in range(len(A1)):
    A1t[i].append(B1t[i][0])


ans12 = library.gauss_jordan(A1t)

print(ans11,ans12)
