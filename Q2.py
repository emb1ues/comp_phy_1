import library
from copy import deepcopy
import numpy as np


a,b,c,d,e,f = np.loadtxt("mat2.txt" , unpack = True)
A2=[]
for i in range(len(a)):
    A2.append([a[i],b[i],c[i],d[i],e[i],f[i]])

B2 = [[-5/3],[2/3],[3],[-4/3],[-1/3],[5/3]]

A2t = deepcopy(A2)
B2t = deepcopy(B2)

library.luDecompose(A2t , len(A2t))
library.forwardBackwardSubstitution(A2t,B2t)
ans21 = deepcopy(B2t)

# Resetting the matrices
A2t = deepcopy(A2)
B2t = B2t = [-5/3,2/3,3,-4/3,-1/3,5/3]
ans22 = library.jacobi(A2t,B2t, 10**(-4))

print(ans21,ans22)

# Resetting the matrices
A2t = deepcopy(A2)
ans23 = library.Inverse(A2t,tol = 1e-4,xsolvername = "JacobiInv" ,plot = True)

# Resetting the matrices
A2t = deepcopy(A2)
ans24 = library.Inverse(A2t,tol = 1e-4,xsolvername = "Gauss-Seidel",plot=True)

# Resetting the matrices
A2t = deepcopy(A2)
ans25 = library.Inverse(A2t,tol = 1e-4,xsolvername = "ConjugateGrad")

print(ans23,ans24,ans25)
