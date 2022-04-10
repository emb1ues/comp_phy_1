from pickle import FALSE
from cv2 import GaussianBlur
from matplotlib import pyplot as plt
import numpy as np
import math

def print_matrix(Matrix):
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            print(Matrix[i][j], end=", ")
        print("")

def matrix_multiplication(matrix1, matrix2):
    M1_cross_M2 = []
    for i in range(len(matrix1)):
        M1_cross_M2.append([])
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            sum = 0
            for k in range(len(matrix2)):
                sum = sum + (matrix1[i][k] * matrix2[k][j])
            M1_cross_M2[i].append(sum)
    return M1_cross_M2        

def jacobi(matrix ,b ,tol =1e-4, plot =False):
    a=1
    aarr=[]
    karr=[]
    p=1
    X = []
    X1= []
    for i in range(len(matrix)):
        X.append(0)
        X1.append(0)

    while(a>tol):
       
        a = 0
        for l in range(len(X)):
            X[l] = X1[l]
                
        for i in range(len(matrix)):
            sum = 0
            for j in range(len(matrix)):
                if( i!=j):
                    
                    sum += matrix[i][j]*X[j] 
            
            X1[i] = (b[i]-sum)/matrix[i][i]
        for j in range(len(X)):
            a += (X1[j]-X[j])**2
            
        a = a**(1/2)
        aarr.append(a)
        karr.append(p)
        p += 1
        
    if(plot==True):
        plt.xlabel('iteration')
        plt.ylabel('residual')
        plt.plot(karr,aarr)
        plt.show()

    return X1
        



def gauss_sidel(matrix ,b ,tol=1e-4 ,plot=False):
    a=1
    aarr=[]
    karr=[]
    p=1
    X = []
    X1= []
    for i in range(len(matrix)):
        X.append(0)
        X1.append(0)
    
    while(a>tol):
        a=0
        for l in range(len(X)):
            X1[l] = X[l]
                
        for i in range(len(matrix)):
            sum = 0
            for j in range(len(matrix)):
                if( i!=j):
                    sum += matrix[i][j]*X[j]         
            
            X[i] = (b[i]-sum)/matrix[i][i]
         
        for j in range(len(X)):
            a += (X1[j]-X[j])**2
        a = a**(1/2)
        aarr.append(a)
       
        karr.append(p)
        p += 1
    if(plot==True):
        plt.xlabel('iteration')
        plt.ylabel('residual')
        plt.plot(karr,aarr)
        plt.show()

    return X
        
        
def ConjGrad(A,b,x = None, tol = 1e-4, max_iter = 1000, plot= False):
    n = len(A)
    if x is None: x = np.ones(n)
    r = b - np.dot(A,x)
    d = r
    count = 0
    while (np.dot(r,r)>tol and count<max_iter):
        rn = np.dot(r,r)
        a = (rn)/(np.dot(d,np.dot(A,d)))
        x += a*d
        r -= a*np.dot(A,d)

        b = np.dot(r,r)/rn
        d = r + b*d
        count += 1
    return x



def Inverse(matrix,tol,plot = False ,xsolvername = "JacobiInv"):
    if(xsolvername == "JacobiInv"): xsolver = jacobi
    if(xsolvername == "Gauss-Seidel"): xsolver = gauss_sidel
    if(xsolvername == "ConjugateGrad"): xsolver = ConjGrad
    
    I = np.identity(len(matrix))
    Inv = np.zeros((len(matrix),len(matrix)))
    for i in range(len(matrix)):
        Inv[:,i] = xsolver(matrix, I[i], tol=tol,plot= plot)

    return Inv   


def polynomial_fit(x, y , u,n):
    A = np.zeros((n,n))
    B = np.zeros(n)
    for i in range(len(A)):
        for j in range(len(A)):
            sum =0 
            for k in range(len(x)):
                sum  += x[k]**(i+j)/u[k]**2
            A[i][j] = sum
    for i in range(len(B)):
        sum =0
        for j in range(len(x)):
            sum += x[j]**(i)*y[j]/u[j]**2
        B[i] = sum
    X = gauss_sidel(A,B)
    fit = []
    for i in range(len(x)):
        fit.append(math.e**(X[0]+X[1]*x[i]))

    Ainv = Inverse(A)
    

    return X  , Ainv 

def partial_pivot(matrix):
    j = 1
    for i in range(len(matrix)):
        if i != len(matrix) - 1 and j == 1:
            if matrix[i][i] == 0:
                for l in range(i, len(matrix)):
                    if matrix[l][i] != 0:
                        for k in range(len(matrix[0])):
                            matrix[l][k], matrix[i][k] = matrix[i][k], matrix[l][k]
                            j = 0

        elif j == 1:
            if matrix[i][i] == 0:
                for k in range(len(matrix) + 1):
                    matrix[i - 1][k], matrix[i][k] = matrix[i][k], matrix[i - 1][k]
                    j = 0

    return matrix

#def gauss_jordan(A):
#    for i in range(len(a)):
#        a = partial_pivot(a)
#        pivot = a[i][i]
#        for j in range(len(a[0])):
#            a[i][j] = a[i][j] / pivot
#        for k in range(len(a)):
#            if k != i:
#                ratio = a[k][i]
#                for j in range(len(a[0])):
#                    a[k][j] = a[k][j] - ratio * a[i][j]
#
#    return a    


# Eigenvalues Calculators

# Function which calculates the largest eigenvalue using power method
def PowerMethodCalc(A, x, tol = 1e-4):
    oldEVal = 0 # Dummy initial instance
    eVal = 2

    while abs(oldEVal-eVal)>tol:
        x = np.dot(A,x)
        eVal = max(abs(x))
        x = x/eVal

        oldEVal=eVal

    return eVal,x


# Wrapper function which allows us to get multiple eigenvalues
def EigPowerMethod(A, x=None, n=1, tol = 1e-4):
    if x is None: x = np.ones(len(A))
    eig = []
    eigvector = []
    E,V = PowerMethodCalc(A,x,tol)
    eigvector.append(V)
    eig.append(E)
    if(n>1):
        iter = n-1
        while iter > 0:
            V = V/np.linalg.norm(V)
            V = np.array([V])
            A = A - E*np.outer(V,V)
            E,V = PowerMethodCalc(A,x,tol)
            eig.append(E)
            eigvector.append(V)
            iter -= 1
    return eig ,eigvector

#Jacobi Method for eigenvalues (Given's Rotation)
def JacobiEig(A):
    n = len(A)
    # Find maximum off diagonal value in upper triangle
    def maxfind(A):
        Amax = 0
        for i in range(n-1):
            for j in range(i+1,n):
                if (abs(A[i][j]) >= Amax):
                    Amax = abs(A[i][j])
                    k = i
                    l = j
        return Amax,k,l

    def GivensRotate(A, tol = 1e-4, max_iter = 5000):
        max = 4
        iter = 1
        while (abs(max) >= tol and iter < max_iter):
            max,i,j = maxfind(A)
            if A[i][i] - A[j][j] == 0:
                theta = math.pi / 4
            else:
                theta = math.atan((2 * A[i][j]) / (A[i][i] - A[j][j])) / 2

            Q = np.eye(n)
            Q[i][i] = Q[j][j] = math.cos(theta)
            Q[i][j] = -1*math.sin(theta)
            Q[j][i] = math.sin(theta) 
            AQ = matrix_multiplication(A,Q)

            # Q inv = Q transpose
            Q = np.array(Q)
            QT = Q.T.tolist()

            A = matrix_multiplication(QT,AQ)
            iter += 1
        return A
    sol = GivensRotate(A)
    return np.diagonal(sol)


def DFT(x):
    """
    Function to calculate the 
    discrete Fourier Transform 
    of a 1D real-valued signal x
    """

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    X = np.dot(e, x)
    
    return X




#####################################################################################




def file_to_matrix(matrix_file):
    Matrix = []
    with open(matrix_file) as file:
        M_row = file.readlines()
    for line in M_row:
        Matrix.append(list(map(lambda i: int(i), line.split(" "))))
    return Matrix

def print_matrix(Matrix):
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            print(Matrix[i][j], end=", ")
        print("")


def augment(matrix1, matrix2, forright=False):
    for i in range(len(matrix1)):
        if (type(matrix2[i]) == int or type(matrix2[i]) == float):
            if forright:
                matrix1[i].insert(0, matrix2[i])
            else:
                matrix1[i].append(matrix2[i])
        else:
            if forright:
                matrix1[i].insert(0, matrix2[i][0])
            else:
                matrix1[i].append(matrix2[i][0])
    return matrix1

def file_opener(file_name):
    Matrix = []
    with open(file_name) as file:
        M_string = file.readlines()
    for line in M_string:
        Matrix.append(list(map(lambda i: float(i), line.split(" "))))
    return Matrix

def unaugment(matrix):
    vector = [ 0 for i in range(len(matrix))]
    for i in range(len(matrix)):
        vector[i] = matrix[i].pop(-1)
    return matrix, vector

def partial_pivot(matrix):
    j = 1
    for i in range(len(matrix)):
        if (i!= len(matrix)-1 and j ==1):
            if (matrix[i][i] == 0):
                for l in range(i,len(matrix)):
                    if(matrix[l][i] != 0):
                        for k in range(len(matrix[0])):
                            matrix[l][k] ,matrix[i][k] = matrix[i][k],matrix[l][k]
                            j = 0  
                    
        elif( j == 1):
            if (matrix[i][i] == 0):
                for k in range(len(matrix)+1):
                    matrix[i-1][k] ,matrix[i][k] = matrix[i][k],matrix[i-1][k]
                    j = 0 

def gauss_jordan(a):
    for i in range(len(a)):
        partial_pivot(a)
        pivot = a[i][i]
        for j in range(len(a[0])):
            a[i][j] = a[i][j]/pivot
        for k in range(len(a)):
            if (k!=i):
                ratio  = a[k][i]
                for j in range(len(a[0])):
                    a[k][j] = a[k][j] - ratio*a[i][j]
    b = []
    for i in range(len(a)):
        b.append(a[i][-1])
    return b

def inverse_gd(a):
    
    for i in range(len(a)):
        vector = [0] * i + [1] + [0] * (len(a) - i)
        new_matrix = augment(a , vector )
    inverse = gauss_jordan(new_matrix)
    inverse_matrix = []
    for i in range(len(a)):
        inverse_matrix.append([])    

    for i in range(len(a)):
        matrix1 , matrix2 = unaugment(inverse)
        inverse_matrix = augment(inverse_matrix ,matrix2, True)
    return(inverse_matrix)

def matrix_multiplication( matrix1 , matrix2):
    M1_cross_M2 = []
    for i in range(len(matrix1)):
        M1_cross_M2.append([])
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            sum = 0
            for k in range(len(matrix2)):
                sum  = sum + (matrix1[i][k]*matrix2[k][j])
            M1_cross_M2[i].append(sum)  
    #print_matrix(matrix1)
    #print_matrix(matrix2)                    
    return M1_cross_M2

def LUdecomposition(A):
    for level in range(len(A)):  # diagonally which stage the algorithm is in
        for col in range(level, len(A)):  # the col for the Upper tri. matrix
            summation = 0
            for sum_term in range(0, level):  # the summation
                summation += A[sum_term][col] * A[level][sum_term]
            A[level][col] = A[level][col] - summation
        for row in range(level, len(A)):  # the row for the lower tri. matrix
            summation = 0
            for sum_term in range(0, level):  # the summation
                summation += A[sum_term][level] * A[row][sum_term]
            if row != level:
                A[row][level] = (A[row][level] - summation) / A[level][level]

def luDecompose(A, n):
    for i in range(n):
        # Upper Triangle Matrix (i is row index, j is column index, k is summation index)
        for j in range(i,n):
            # Summation part
            sum = 0
            for k in range(i):
                if(i==k):
                    sum += A[k][j]  # Since diagonal elements of Lower matrix is 1
                else:
                    sum += A[i][k]*A[k][j]
        
            A[i][j] = A[i][j] - sum
        
        # Lower Triangle Matrix (j is row index, i is column index, k is summation index)
        for j in range(i+1,n):
            # Summation part
            sum = 0
            for k in range(i):
                if(j==k):
                    sum += A[k][i]  # Since diagonal elements of Lower matrix is 1
                else:
                    sum += A[j][k]*A[k][i]
            A[j][i] = (A[j][i] - sum)/A[i][i]

    return A        

def forwardBackwardSubstitution(A,B):
    m = len(A)
    n = len(B[0])
    #Forward Substitution
    for j in range(n):
        for i in range(m):
            sum = 0
            for k in range(i):
                sum += A[i][k]*B[k][j]
            B[i][j] = B[i][j] -  sum

    #Backward Substitution
    for j in range(n):
        for i in range(m-1,-1,-1):
            sum = 0
            for k in range(i+1,m):
                sum += A[i][k]*B[k][j]
            B[i][j] = (B[i][j] - sum)/A[i][i]


def decomposed_solver(matrix):
    n = len(matrix)
    # solving Ly =B
    y = [[0]for i in range(n)]
    k=0
    for i in range(n) :
        sum =0
        if(i>0):
            for j in range(k+1):
                sum += matrix[i][j]*y[j][0]    
        y[i][0] = matrix[i][n] - sum
        k= k+1
    #solving Ux = y
    x = [[0]for i in range(n)]
    k=0
    for i in reversed(range(n)) :
        sum =0
        if (i<n-1):
            for j in range(k+1):
                sum += matrix[i][n-j-1]*x[n-j-1][0]
        x[i][0] = (y[i][0]- sum)/matrix[i][i]
        k= k+1

    return x


def inverse_usingLU(matrix):
    n = len(matrix)
    for i in range(n):
        vector = [0] * i + [1] + [0] * (n- i-1)
        whole_matrix = augment(matrix , vector )   
    for i in range(n):       
        partial_pivot(matrix)

    pivot_i =[]
    for i in reversed(range(n)):
        matrix1 , vec = unaugment(whole_matrix )
        if (i == 0):
            pivot_i = vec
        else:
            pivot_i = augment(pivot_i ,vec) 
    new_matrix = matrix1.copy()
    for i in range(n):
        vector = [0] * i + [1] + [0] * (n- i-1)
        whole_matrix = augment(matrix , vector )   
    for i in range(n):       
        partial_pivot(matrix)

    solution =[]
    for i in reversed(range(n)):
        print_matrix(whole_matrix)
        whole_matrix , vec = unaugment(whole_matrix )
        combined_matrix = augment(new_matrix,vec)
        if (i == 0):
            solution = decomposed_solver(combined_matrix)
        else:
            solution = augment(solution  , list (map(lambda i:i[0] , decomposed_solver(combined_matrix))))

    print_matrix(solution)

def fun_prime(f, h=10 ** -4):

    """
    Returns a function for numerical derivative of f
    """

    def df_dx(x, h=h):
        return (f(x + h) - f(x - h)) / (2 * h)

    return df_dx

def derivative(f , x):
    return (f(x+0.000001)-f(x-0.000001))/0.000002

def bracketing(function ,a, b):
    i =0
    while(function(a)*function(b) > 0 and i<15):
        eta = 1.5
        if (function(a) >0):
            a = a-eta*(b-a)
        else:
            b = b+eta*(b-a)
        i+=1
    return bisection_method(function , a,b) , false_method(function ,a,b)

def bisection_method(function ,a,b):
    iteration , error = [] ,[]
    i=0
    c=0
    while(i<50 and abs(a-b) > 0.000001 ):
        l=c
        c = (a+b)/2
        if (function(a)*function(c)==0):
            return c
        elif (function(a)*function(c)<0):
            b=c
        else:
            a=c
        error.append(abs(l-c))
        iteration.append(i)
        i+=1  
    #plt.title('Bisection_method')
    #plt.xlabel('iteration number')
    #plt.ylabel('absolute error')     
    #plt.plot(iteration,error, marker = '.')
    #plt.show()
    return a   

def false_method (function ,a,b):
    iteration , error = [] ,[]
    i=0
    c=0
    while(i<50 and abs(a-b) > 0.000001):
        l=c
        c = b-((b-a)*function(b)/(function(b)-function(a)))
        if(function(a)*function(c)==0):
            return c
        elif(function(a)*function(c)<0):
            b=c
        else:
            a=c
        error.append(abs(l-c))
        iteration.append(i)
        i+=1   
    #plt.title('Falsi_method')
    #plt.xlabel('iteration number')
    #plt.ylabel('absolute error')    
    #plt.plot(iteration,error , marker = '.')
    #plt.show()        
    return c   

def newton_raphson(function,x):
    i=1
    x0 = x
    x1 = 0
    iteration , error = [] ,[]
    while(i<50 and abs(x1-x0) > 0.000001):
        x1=x0
        x0 = x0-(function(x0)/derivative(function,x0))
        error.append(abs(x1-x0))
        iteration.append(i)
        i+=1
    #plt.title('Newton Raphson')
    #plt.xlabel('iteration number')
    #plt.ylabel('absolute error')    
    #plt.plot(iteration,error,marker = '.')
    #plt.show()      
    return x0    

def polynomial_generator(a):
    return lambda x: sum(a_i * pow(x, i) for i, a_i in enumerate(a))


def synthetic_division(coefficients, divisor):
    coefficients = coefficients[::-1]
    quotient = []
    quotient.append(coefficients[0])
    for i in range(1, len(coefficients)):
        quotient.append(coefficients[i] + (divisor * quotient[-1]))
    return quotient[0:-1][::-1]


def laguerre(f, order, guess, ϵ=10 ** -4):
    df_dx = fun_prime(f)
    d2f_dx2 = fun_prime(df_dx )

    if (f(guess) == 0):
        return guess
    else:    
        guess_old = guess + 1  # just to start
        while abs(guess_old - guess) > ϵ:
            G = df_dx(guess) / f(guess)
            H = G * G - (d2f_dx2(guess) / f(guess))
            last_term = ((order - 1) * (order * H - G)) ** (1 / 2)
            sum = G + last_term
            diff = G - last_term
            a = order / (
                sum * (abs(sum) > abs(diff))+ diff * (abs(sum) < abs(diff))
            )  # this is less readable but its faster than an if else
            guess_old = guess
            guess -= a
        return guess


def roots_from_laguerre(coefficients):
    ϵ=10 ** -4 
    guess=1.0
    order = len(coefficients) - 1
    roots = []
    for i in range(order):
        f = polynomial_generator(coefficients)
        root = laguerre(f, order, guess, ϵ)
        coefficients = synthetic_division(coefficients, root)
        roots.append(root)
    return roots    


from random import random

def midpoint(f,a,b,N):
    h = (b-a)/N
    sum =0
    for i in range (N):
        x = (2*a + (2*i+1)*h)/2
        sum += f(x)*h
    return sum  

def trapezoidal(f,a,b,N):
    h = (b-a)/N
    sum =0
    for i in range (1,N):
        x = a+i*h
        sum += f(x)*h
    sum += (f(a)+f(b))*h/2   
    return sum

def simpson(f,a,b,N):
    h = (b-a)/N
    sum =0
    for i in range (1,N):
        x = a+i*h
        weight = 4 if i % 2 else 2
        sum += f(x)*weight
    sum += (f(a)+f(b))
    sum *= h/3
    return sum

def montecarlo(f,a,b,N):
    h = (b-a)/N
    sum =0
    for i in range (N):
        sum += f(a + (b-a)*random())*h
    return sum          


def expliciteuler(n , h ,f , x1 ,y0 ):
    #adding first elements
    y=[y0]
    x=[x1]
    y1=y0
    x0=x1
    # calculating different points
    for i in range(n):
        y1 += h*f(y1,x0)
        x0 += h
        y.append(y1)  #appending reslts in array to plot
        x.append(x0)

    return x,y


def rk4(n, h ,f , x1 ,y0):
    """
    n: no of steps
    h: step length
    f: dy/dx
    x1, y0: initial conditions
    """
    #adding first elements
    y=[y0]
    x=[x1]
    y1=y0
    x0=x1
    #defining k1,k2,k3,k4 for calculating points using simpson fromula 
    for i in range(n):
        k1 = h*f(y1,x0)
        k2 = h*f(y1+(k1/2),x0+(h/2))
        k3 = h*f(y1+(k2/2),x0+(h/2))
        k4 = h*f(y1+k3,x0+h)
        y1 += (k1+2*k2+2*k3+k4)/6
        x0 += h
        y.append(y1) #appending reslts in array to plot
        x.append(x0)
    
    # loop for running backwards 
    y1=y0
    x0=x1
    for i in range(n):
        k1 = h*f(y1,x0)
        k2 = h*f(y1-(k1/2),x0-(h/2))
        k3 = h*f(y1-(k2/2),x0-(h/2))
        k4 = h*f(y1-k3,x0-h)
        y1 -= (k1+2*k2+2*k3+k4)/6
        x0 -= h
        y.append(y1)
        x.append(x0)
    
    return x,y

def shooting_method(n,
    dz_dx, BV1, BV2,dx, method=rk4, limit=10
):

    count = 0
    guess1 = [BV1[0], 1]
    guess2 = [BV1[0], -1]
    if BV2[1] != 0:
        y1 = [0]
    else:
        y1 = [1]
    x1 = [0]
    while (
        abs(y1[-1] - BV2[1]) >= 10 ** -13
        and count < limit
    ):
        if count == 0:
            guess = guess1.copy()
        elif count == 1:
            guess1.append(y1[-1])
            guess = guess2.copy()
        else:
            if count == 2:
                guess2.append(y1[-1])
            else:
                guess1[2] = y1[-1]
            # generating new guess
            guess = guess1[1] + (guess2[1] - guess1[1]) * (BV2[1] - guess1[2]) / (guess2[2] - guess1[2])
            guess1[1] = guess
            guess = guess1
        # using rk4 to calculate
        x1, z1 = method(int(2*n+1),
            dx/2,
            dz_dx,
            guess[0], guess[1],
        )
        x1 = list(map(lambda x: round(x, 6), x1))

        def dy_dx(y, x):
            return z1[x1.index(round(x, 6))]
        x1, y1 = method(int(n),
            dx,
            dy_dx,
            BV1[0],BV1[1])
        count += 1
    return x1, y1
