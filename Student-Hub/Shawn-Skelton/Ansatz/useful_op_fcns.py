import pennylane as qml
from pennylane import numpy as np

def P_STRING(A):
    """returns the kron product of an array of Pauli matrices. Input: a numpy array indexed [pauli term, row, column]. For example np.array([sY, sY, sY]). Output: a numpy array with dimension 2^{number of Pauli matrices}, the kroneker product of arguement matrices. Eg $\sigma_Y\otimes\sigma_Y\otimes\sigma_Y$"""
    breadth=len(A)
    P_string=np.array([1])
    for i in range(0,breadth):
        P_string=np.kron(P_string, A[i, :, :])
        
    return P_string

def COMMUTATOR(A, B):
    """
    does what it sounds like. A, B are square numpy arrays with same dimension, returns the commutator as a numpy array
    """
    return A@B-B@A

##   
def CGATE(G, target=2, proj0=np.array([[1, 0], [0, 0]]), proj1=np.array([[0, 0], [0, 1]])):
    """
    Compute the controlled version of gate G. 
    Input: 
    G: a square numpy array, the gate we want to a controlled version of; 
    target: determines whether the target is the 1st or the second subsystem
    proj0, proj1: square numpy arrays, the projectors determining which part of the Hilbert space G is applied in. default is $proj0=\bra{0}\ket{0}$, $proj1=\bra{1}\ket{1}$
    Output: product $proj0\otimes I+proj1\otimes G$
    """
    g=len(G)
    if target==1:
        return np.kron(np.identity(g), proj0)+np.kron(G, proj1)
    else:
        return np.kron(proj0, np.identity(g))+np.kron(proj1, G)

def MATRIX_CHECK(A, tol=10**(-16), rtn="binary"):
    """
    checks to see if A=0, where "0" is machine precision
    inputs:
    A: the matrix we want to solve
    rtn: string controlling the return, default is "binary". any other string will change the return to be A, or 0
    output: either a binary value (0, 1) or the matrix
    """
    flag=1
    if np.all(abs(np.real(A))<tol)==True and np.all(abs(np.imag(A))<tol)==True:
     flag=0
     
    if rtn!="binary":
        if flag==0:
            a=len(A)
            return np.zeros([a, a])
        else:
            return A
    return flag
    
    