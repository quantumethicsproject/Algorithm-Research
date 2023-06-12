# -*- coding: utf-8 -*-
"""
Created on Tue May 23 20:45:21 2023

@author: Shawn Skelton

Code to simulate adiobatically assisted VQE
adapted from Pennylane tutorial: https://pennylane.ai/qml/demos/tutorial_vqe
"""
import pickle
from pennylane import numpy as np
import pennylane as qml
from pennylane import qchem
import time
#import matplotlib.pyplot as plt

####some useful definitions###
filename='H_net_data.pkl'
E_fci = -1.136189454088 #The GSE to compare, if known. 

##Create arrays to store the energy and angle values for each step of the iteration
sangle=[]
senergy =[]
sn=[]
ts=[]

ssteps=8

energy=[]
angle=[]
n=[]
t=[]

###build the molecule Hamiltonian: start simple, so Hydrogen
def H_mol():
    symbols= ["H", "H"]
    coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    H0=qml.Hamiltonian(np.ones(qubits), [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2), qml.PauliX(3)])
    electrons=2
    return H, H0, electrons, qubits

def H2O_mol():
    symbols = ["H", "O", "H"]
    coordinates = np.array([-0.0399, -0.0038, 0.0, 1.5780, 0.8540, 0.0, 2.7909, -0.5159, 0.0])
    charge=0
    multiplicity=1
    basis_set = "sto-3g"
    electrons = 10
    orbitals = 7
    core, active = qchem.active_space(electrons, orbitals, active_electrons=4, active_orbitals=4)
    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        charge=charge,
        mult=multiplicity,
        basis=basis_set,
        active_electrons=4,
        active_orbitals=4,
    )
    
    H0=qml.Hamiltonian(np.ones(qubits), [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2), qml.PauliX(3), qml.PauliX(4), qml.PauliX(5), qml.PauliX(6), qml.PauliX(7)])
    electrons = 4
    return H, H0, electrons, qubits


###define the circuit. We'll use the Hartree Fock state to begin, plus an excitation term
H, H0, electrons, qubits=H_mol()
dev = qml.device("default.qubit", wires=qubits)
##Penny lane has a function for the Hartree-Fock state
hf = qml.qchem.hf_state(electrons, qubits)
#we'll start out with parameter value theta=0
theta = np.array(0.0, requires_grad=True)

##Then we'll define an operator acting on it - again Pennylane has an inbuilt function for this
def circuit(param, wires):
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

def scircuit(param, wires):
    qml.BasisState(np.zeros(len(hf)), wires=wires)
    
##Now we get to define the cost function, where here the measured operation is energy
@qml.qnode(dev, interface="autograd")
def cost_fnAA(param):
    circuit(param, wires=range(qubits))
    return qml.expval((1-s)*H0+s*H)    

@qml.qnode(dev, interface="autograd")
def cost_fn(param):
    circuit(param, wires=range(qubits))
    return qml.expval(H)

###Classical optimization: use gradient descent (others available in Pennylane)
def VQE(th0,cost_fc=cost_fn):
    """
    Function to run VQE. Arguement are the cost function, the initial parameter value

    Returns
    -------
    None.

    """
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    max_iterations = 600 
    conv_tol = 1e-06
    
    energy=[]
    angle=[]
    theta=th0
    
    t0r=time.perf_counter()
    
    for n in range(max_iterations):
        ##actually runs each optimization step and returns new parameters
        theta, prev_energy= opt.step_and_cost(cost_fc, theta)
        energy.append(cost_fn(theta))
        angle.append(theta)  
        conv = np.abs(energy[-1] - prev_energy)
           
        if conv <= conv_tol:
            break
        
    #print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
    #print("\n" f"Optimal value of the circuit parameter = {angle[ -1]:.4f}")
    
    t1r=time.perf_counter()

    return n, angle[-1], energy[-1], t1r-t0r
    
###Now run the VCA simulation with adiobatic evolution:
sarray=np.linspace(0, 1, ssteps) 
sangle=[]
senergy =[]
th0=np.array(0.0, requires_grad=True)

for sind, s in enumerate(sarray):
    t0s=time.perf_counter()
    print(s)
    n, theta, E, t=VQE(th0,cost_fc=cost_fnAA)
    senergy.append(E)
    sangle.append(theta)
    sn.append(n)
    ts.append(t)
    th0=theta


###Save the high view data in a pickle file: how many total iterations, what was the final energy result, what was the time?
data={'n':[], 'angle':[], 'energy':[], 't': [], 's': [], 'ns':[], 'sangle':[], 'senergy':[], 'ts':[]}
data['s']=ssteps
data['ns']=sn
data['sangle']=sangle
data['senergy']=senergy
data['ts']=ts


###Now compare to standard VCE: how does the time and total number of iterations compare?

theta = np.array(0.0, requires_grad=True)
n, theta,E, t=VQE(np.array(0.0, requires_grad=True), cost_fn )
energy.append(E)
angle.append(theta)

data['n'].append(n)
data['angle'].append(angle[-1])
data['energy'].append(energy[-1])
data['t'].append(t)


with open(filename, 'wb') as manage_file:
    pickle.dump(data, manage_file)
    
###Plots: Now moved to a seperate file
# fig=plt.figure()
# fig.set_figheight(5)
# fig.set_figwidth(12)

# Add energy plot on column 1
# ax1 = fig.add_subplot(121)
# ax1.plot(sarray, senergy, "go", ls="dashed")
# #ax1.plot(sarray, np.full(ssteps, E_fci), color="red")
# ax1.set_xlabel("Optimization step", fontsize=13)
# ax1.set_ylabel("Energy (Hartree)", fontsize=13)
# ax1.text(0.5, -1.1176, r"$E_\mathrm{HF}$", fontsize=15)
# ax1.text(0, -1.1357, r"$E_\mathrm{FCI}$", fontsize=15)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# #Add angle plot on column 2
# ax2 = fig.add_subplot(122)
# ax2.plot(sarray, sangle, "go", ls="dashed")
# ax2.set_xlabel("Optimization step", fontsize=13)
# ax2.set_ylabel("Gate parameter $\\theta$ (rad)", fontsize=13)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.subplots_adjust(wspace=0.3, bottom=0.2)
# plt.show()

###TO do
##understand step_and_cost in Pennylane: DONE, just a dcn that neatly runs the optimizer. The args shouldn't change for us
##Add adiobatic loop and change the cost function (leave the molecule Hamiltonian as is) DONE
##understand why the looping thing in s is returning n=0 cases DONE 
#(because the n=0 step starts off with (a) energy guess computed with s-1 theta and s hamiltonain, and then finds a new energy guess. So this is actually a good sign)
##pick a new test hamiltonian DONE, (ish) water seems too large to work
##add in noise. Q: how to I pick types of noise to consider? 
#https://pennylane.ai/blog/2021/05/how-to-simulate-noise-with-pennylane/
##pick a good standard of data collection and plots: DONE
##Pick a standard ansatz option 
#Hamiltonian choices
#choice of adiobatic 'easy' guess: try using the HF state and make the ansatz the comp basis state??