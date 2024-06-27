# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:45:13 2023

@author: Shawn Skelton
"""

from pennylane import numpy as np
import pennylane as qml
from pennylane import qchem
import time
import matplotlib.pyplot as plt
import pickle
import os.path

#Hdata = qml.data.load("qchem", molname="H2", basis="STO-3G", bondlength=1.1)
#print(Hdata['fci_energy'])

part = qml.data.load("qchem", molname="H2", basis="STO-3G", bondlength=1.1, attributes=["molecule", "fci_energy"])[0]
part.molecule

d=1
ctol=1.6*10**(-3)
mit=600
qubits=4
electrons=2

hf = qml.qchem.hf_state(electrons, qubits)
dev=qml.device('default.qubit', wires=qubits)

###build the molecule Hamiltonian: start simple, so Hydrogen
def H_mol(a):
    symbols= ["H", "H"]
    coordinates = np.array([0.0, 0.0, -a, 0.0, 0.0, a])
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    H0=qml.Hamiltonian(np.ones(qubits), [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2), qml.PauliX(3)])
    electrons=2
    return H, H0, electrons, qubits


##Then we'll define an operator acting on it - again Pennylane has an inbuilt function for this
def circuit(param, wires):
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    
def U_ENT(wires):
    sigmaX=np.array([[0,1],[1, 0]])
    sigmaZ=np.array([[1,0],[0, -1]])
    U=np.array(sigmaZ@sigmaX)
    ###CNOTs to each, seems to work okay
    # qml.CNOT([len(wires)-1, 0])
    qml.ControlledQubitUnitary(U, 1, 0)
    qml.ControlledQubitUnitary(U, 0, 2)
    qml.ControlledQubitUnitary(U, 1,3)
    # for i in range(len(wires)-1):
    #     qml.CNOT([i, i+1])
    
def kandala_circuit(param, wires, d):
    ###indexing looks a bit fucked up bc its a 1d list (should be easier for pennylane basics)
    ###as a 3d array the indexing would be [d iteration, qubit number, R number]
    ###given $\theta_{j i}^q$ $j\in{1, 2, 3}$
    ###as a 1d list the sequence is [\theta_{00}^0, \theta_{10}^0, \theta_{20}^0, \theta_{01}^0...]
    ###all zeros state
    qml.BasisState(np.zeros(len(wires)), wires=wires)
    ###apply the first set of Euler rotations, without RZ terms
    indtrack=0
    for q in range(len(wires)):
        qml.RX(param[indtrack], wires=[q])
        qml.RZ(param[indtrack+1], wires=[q])
        indtrack=indtrack+2
    
    for i in range(d):
        U_ENT(wires)
        for q in range(len(wires)):
            #print('qubit',q,  indtrack)
            qml.RZ(param[indtrack], wires=[q])
            qml.RX(param[indtrack+1], wires=[q])
            qml.RZ(param[indtrack+2], wires=[q])
            indtrack=indtrack+3
            #print(indtrack)
        #print(q, indtrack)
        
@qml.qnode(dev, interface="autograd")
def kandala_cost_fcn(param):
    kandala_circuit(param, range(qubits), d)
    return qml.expval(H)
    
@qml.qnode(dev, interface="autograd")
def cost_fn(param):
    circuit(param, wires=range(qubits))
    return qml.expval(H)

###Classical optimization: use gradient descent (others available in Pennylane)
def VQE(th0,cost_fc=cost_fn,  systsz=qubits, max_iterations=mit, conv_tol=ctol, gradDetect=False):
    """
    Function to run VQE. Arguement are the cost function, the initial parameter value

    Returns
    -------
    None.

    """
    
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    #opt = qml.SPSAOptimizer(maxiter=max_iterations, c=0.15, a=0.2)
   
    energy=[]
    angle=[]
    grad=[]
    theta=th0
    
    t0r=time.perf_counter()
    prev_g=0
    for n in range(max_iterations):
        ##actually runs each optimization step and returns new parameters
        theta, prev_energy, g= opt.step_and_cost(cost_fc, theta)
        # if np.any(abs(prev_g-g))<np.exp(-systsz):
        #     print('warning, barren plateau at ', n, 'th iteration' )
        # prev_g=g
        
        grad.append(g)
        energy.append(cost_fn(theta))
        angle.append(theta)  
        conv = np.abs(energy[-1] - prev_energy)
           
        if conv <= conv_tol:
            break
        
    #print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
    #print("\n" f"Optimal value of the circuit parameter = {angle[ -1]:.4f}")
    
    t1r=time.perf_counter()
    if gradDetect==True:
        plt.plot(np.linspace(0, n, n+1), grad,  marker='X', linestyle='dashed',)
        plt.legend()
   
    return n,energy[-1], angle[-1],  t1r-t0r
    
def kandala_VQE(param0, d, cost_fc=kandala_cost_fcn, systsz=qubits, max_iterations=mit, conv_tol=ctol,gradDetect=False):
    """
    Function to run VQE. Arguement are the cost function, the initial parameter value

    """
    #
    
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    #opt = qml.SPSAOptimizer(maxiter=max_iterations, c=0.01, a=0.602)
    
    energy=[]
    thetas=param0
    t0r=time.perf_counter()
    prev_g=np.zeros([len(param0)])
    
    for n in range(max_iterations):
        ##actually runs each optimization step and returns new parameters
        thetas, prev_energy, g= opt.step_and_cost(cost_fc, thetas)
        # print(abs(prev_g-g))
        # print(np.exp(-systsz))
        # if abs(prev_g-g).any<np.exp(-systsz):
        #     print('warning, barren plateau at ', n, 'th iteration' )
        prev_g=g
        energy.append(cost_fc(thetas))
        
        ##now there are too many angles to store - instead compute the average change in angles in one iteration
       
        conv = np.abs(energy[-1] - prev_energy)
        
        if conv <= conv_tol:
            break
        
    #print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
    #print("\n" f"Optimal value of the circuit parameter = {angle[ -1]:.4f}")
    
    t1r=time.perf_counter()
    print( t1r-t0r)
    # if gradDetect==True:
    #     plt.plot(np.linspace(0, n, n+1), grad,  marker='X', linestyle='dashed',)
    #     plt.legend()
    return n, energy[-1], thetas, t1r-t0r



    

#n, theta,E, t=VQE(np.array(0.0, requires_grad=True), cost_fn,)
#energy.append(E)
#angle.append(theta)

###build a parameter list for 4 qubits, 3 rotations, d=1 iterations + initial
#params=np.array([theta]*(3*d*qubits+2*qubits))
#n,DeltaTh, E, t=kandala_VQE(params, d, cost_fc=kandala_cost_fcn, gradDetect=False)

interd0=0.1
interd1=4
numpoints=1
iad_array=np.linspace(interd0,interd1, numpoints)
theta0 =np.array(0.1, requires_grad=True)
params0=np.array([theta0]*(3*d*qubits+2*qubits))

###MAIN LOOP: solve the VQE for each interatomic distance, using Pennylane's demo and the Kandala approach
###save data on the number of iterations and time, along with VQE data
##for now, don't save gradients 
KGS=[]
kenergy=[]
kangles=[]
kits=[]
ktimes=[]

energy=[]
angles=[]
its=[]
times=[]

Hams=[]

for a, iad in enumerate(iad_array):
    print('atomic distance', iad)
    H, H0, electrons, qubits=H_mol(iad)
    Hams.append(H)
    GSE=42
    KGS.append(GSE)
    kn, kE, thetas, kt=kandala_VQE(params0, d, cost_fc=kandala_cost_fcn, gradDetect=False)
    print(kn, kE, kt)
    kits.append(kn)
    kenergy.append(kE)
    ktimes.append(kt)
    n, E, theta, t=VQE(theta0, cost_fn,)
    its.append(n)
    energy.append(E)
    angles.append(theta)
    times.append(t)
    print(n, E, t)

data={'GSE': KGS, 'kits':kits, 'kenergy':kenergy, 'kangle':kangles, 'ktimes': ktimes, 'its':its, 'energy':energy, 'angle':angles, 'times': times, 'number_interd': numpoints, 'interatom_d': iad_array, 'init_kparam': params0, 'init_param': theta0, 'Hams': Hams, 'ansatz_depth': d, 'solver':'GD_0.04' , 'max_iterations': mit, 'conv_tol': ctol}

filename='kandala_H2_'+str(numpoints)+'_iads.pkl'


save_path = 'C:/Users/Shawn Skelton/Documents/AAVQE/03_data'
completename = os.path.join(save_path, filename)         


# with open(completename,'wb') as file:
#     pickle.dump(data, file)


##initial point, entangling gates
##into AAVQE