# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 18:11:11 2023

@author: skelt
"""

import numpy as np
from scipy.linalg import expm

X=np.array([[0, 1], [1, 0]])
Y=np.array([[0, -1j], [1j, 0]])
Z=np.array([[1, 0], [0, -1]])
I=np.array([[1, 0], [0, 1]])
p1=np.array([[0, 0], [0, 1]])
p0=np.array([[1, 0], [0, 0]])
CNOT=np.kron(p0, I)+np.kron(p1, X)
CY=np.kron(p0, I)+np.kron(p1, Y)
#print(-1j*expm(1j*np.pi/2*(1/np.sqrt(2)*X+1/np.sqrt(2)*Z)))
#print(np.kron(I, I)@expm(1j*np.pi/2*np.kron(X, X))@np.kron(I, I))
