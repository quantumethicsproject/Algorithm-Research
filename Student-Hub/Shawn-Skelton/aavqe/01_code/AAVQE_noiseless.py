import pennylane as qml
from pennylane import numpy as np
import time
import pickle
import os.path
import matplotlib.pyplot as plt
from problems.H2_problem import H2_PROBLEM

from cost_functions import cost_fnAA
# from problems.useful_fcns import EASY_HAM

from pennylane_cirq import ops as cirq_ops
######################################
molname='H2'
qubits=4
d=1
bdl_array=['0.5']
theta0=np.random.rand(1)
params0=np.random.rand(3*d*qubits+2*qubits) #np.array([theta0]*(3*d*qubits+2*qubits))

dev=qml.device('default.qubit', wires=qubits)
cost_fnAA=qml.QNode(cost_fnAA, dev, interface="autograd")
####DEFINE THE DEVICE AND COST FUNCTION
mit=30
ctol=1.6*10**(-3)

####FUNCTIONS TO MOVE EVENTUALLY####
def AA_VQE(H, H0, param0, d, sits, cost_fc, max_iterations, conv_tol):
        opt = qml.GradientDescentOptimizer(stepsize=0.4)
        
        energy=[]
        svarg=[]
        thetas=param0

        if sits==1:
            s_array=np.array([1])
        else:
            s_array=np.linspace(0, 1, sits, endpoint=True)
        svqe=1
        t0s=time.perf_counter()
        for sind, svq in enumerate(s_array):
            for n in range(max_iterations):
                ##actually runs each optimization step and returns new parameters
                
                thetas, prev_energy, g= opt.step_and_cost(cost_fc, thetas, H,H0, qubits, svqe)
                
                energy.append(cost_fnAA(thetas, H,H0, qubits, svqe))

                conv = np.abs(energy[-1] - prev_energy)
                if conv <= conv_tol:
                    break
                
            #plt.plot(range(n), energy)
        t1s=time.perf_counter()

        return n, energy[-1], thetas, t1s-t0s, svarg, energy

for ind, bdl in enumerate(bdl_array):
    instance=H2_PROBLEM(molname, bdl)
    Hvqe=instance.H
    H0vqe=instance.H0
    
    Etest=cost_fnAA(params0,Hvqe, H0vqe, qubits)
    n, Eguess, thetas, time, svarg, energy=AA_VQE(Hvqe, H0vqe, params0, d, 1, cost_fnAA, mit, ctol)
    
    #AA_VQE(Hvqe, H0vqe, params0, d, 1, cost_fnAA, mit, ctol)
    ###GENERATE INITIAL COEFFICIENTS#####
