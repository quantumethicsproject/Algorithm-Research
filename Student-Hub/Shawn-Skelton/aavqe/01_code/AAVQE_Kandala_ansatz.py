# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 21:05:48 2023

@author: skelt
"""

from pennylane import numpy as np
import pennylane as qml
#from pennylane import qchem
import time
#import matplotlib.pyplot as plt
import pickle
import os.path

import matplotlib.pyplot as plt
from pennylane_cirq import ops as cirq_ops
####constants
ifsave=False
mol='H2'
numpoints=1
d=1
ctol=1.6*10**(-3)
mit=200
ssteps=4



###want to automate eventually:
qubits=4
###stuff for the variance: want an randomized order of magnitude bound for the variance

#theta0 =np.array(0.1, requires_grad=True)
###the shape we want is tensor([...], requires_grad=true)
theta0=np.random.rand(1)
params0=np.random.rand(3*d*qubits+2*qubits) #np.array([theta0]*(3*d*qubits+2*qubits))

electrons=2
#dev=qml.device('default.qubit', wires=qubits)

dev = qml.device("cirq.mixedsimulator", wires=qubits)
GS=[]
Hams=[]

kenergy=[]
kangles=[]
kits=[]
ktimes=[]
kvarg=[]

sangle=[]
senergy =[]
sn=[]
st=[]
sdictvarg={}

sarray=np.linspace(0, 1, ssteps) 

# available_data = qml.data.list_datasets()["qchem"][mol]["STO-3G"]
# bdl_array=available_data[1:2]

bdl_array=np.linspace(-1, 10, 10)


def MOL_H_BUILD(mol, bdl):
    part = qml.data.load("qchem", molname=mol, basis="STO-3G", bondlength=bdl, attributes=["molecule", "hamiltonian", "fci_energy"])[0]
    H=part.hamiltonian
    H0=qml.Hamiltonian(np.ones(qubits)/2, [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2), qml.PauliX(3)])
    #FCI:  Full configuration interaction (FCI) energy computed classically should be in Hatrees
    gsE=part.fci_energy
    return H, H0, gsE

#keyword variables need some functional definition so here's the default settings
#Hdef,H0def, gsEdef=MOL_H_BUILD(mol, bdl_array[0])


def ISING_HAM(sites, J, h):
    """
    Builds a 1D nonperiodic Ising hamiltonian and computes the ground state energy using exact diagonalization
    we assume all J, h>0 and $h_i=h_j=h, J_i=J_j=J$ for simplicity. 
    In math, the $n$ site(qubit) Ising hamiltonian is $H=-J\sum_{i=0}^{n-1} Z_iZ_{i+1}-h\sum_{i=0}^nX_i$
    H_0 is the simple hamiltonian we need for the AAVQE step
    """
    Gset1q=[qml.PauliX(i) for i in range(0, sites)]
    Gset2q=[qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range(0, sites-1)]
    coeffs=np.append(-h*np.ones(sites),-J*np.ones(sites-1))
    
    H0=qml.Hamiltonian(np.ones(sites)/(sites), Gset1q)
    H=qml.Hamiltonian(coeffs, Gset1q+Gset2q)
    
    eigs=np.linalg.eigvals(qml.matrix(H))
    gse=min(np.real(eigs))
    return H, H0, gse

def XX_HAM(sites, lamb):
    """
    Builds a 1D nonperiodic XX hamiltonian (I think a simplification of the Heisenburg model they didn't write correctly) and computes the ground state energy using exact diagonalization
    we assume all \lambda?>0. 
    In math, the $n$ site(qubit) Ising hamiltonian is $H=J\sum_{i=0}^{n-1} X_iX_{i+1}+\lambda\sum_{i=0}^nX_i$
    H_0 is the simple hamiltonian we need for the AAVQE step
    """
    Gset1q=[qml.PauliX(i) for i in range(0, sites)]
    Gset2q=[qml.PauliX(i) @ qml.PauliX(i+1) for i in range(0, sites-1)]
    coeffs=np.append(lamb*np.ones(sites),np.ones(sites-1))
    
    H0=qml.Hamiltonian(np.ones(sites)/(sites), Gset1q)
    H=qml.Hamiltonian(coeffs, Gset1q+Gset2q)
    
    eigs=np.linalg.eigvals(qml.matrix(H))
    gse=min(np.real(eigs))
    return H, H0, gse

Hdef,H0def, gsEdef=XX_HAM(qubits, bdl_array[0])
sdef=1
#####hardware efficient ansatz
def U_ENT(wires):
    """
    we will follow the description in Larocca et al. extrapoloated from Fg. 4 and page 23. 
    ZZ entangling gates are applied to neighbouring qubits, with a lattice rather than periodic set-up
    FOR NOW, USE REGULAR CNOT INSTEAD OF exp(i\pi/2Z\otimesZ)
    """
    sigmaX=np.array([[0,1],[1, 0]])
    sigmaZ=np.array([[1,0],[0, -1]])
    # U=np.array(sigmaX)
    num_qubits=len(wires)
    for j in range(0,num_qubits-1 ):
        #qml.CNOT([j, j+1])
        qml.IsingZZ(np.pi, [j, j+1])
        # qml.ControlledQubitUnitary(U, 1)
    ###CNOTs to each, seems to work okay
    # qml.ControlledQubitUnitary(U, 1, 0)
    # qml.ControlledQubitUnitary(U, 0, 2)
    # qml.ControlledQubitUnitary(U, 1,3)
    
    
def kandala_circuit(param, wires, d):
    ###simplified circuit from http://arxiv.org/abs/1704.05018

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
        
def circuit(param, wires):#
    hf = qml.qchem.hf_state(electrons, qubits)
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

def scircuit(param, wires):
    ###meant to compare AAVQE and VQE for the H2 mol, where a 1param circuit can succeed
    qml.BasisState(np.zeros(len(wires)), wires=wires)
    
####Kandala cost function
@qml.qnode(dev, interface="autograd")
def kandala_cost_fcn(param, H=Hdef):
    kandala_circuit(param, range(qubits), d)
    return qml.expval(H)
    
@qml.qnode(dev, interface="autograd")
def cost_fn(param, H=Hdef):
    circuit(param, wires=range(qubits))
    return qml.expval(H)

@qml.qnode(dev, interface="autograd")
def cost_fnAA(param, H=Hdef, H0=H0def, s=sdef): 
    kandala_circuit(param, range(qubits), d)
    return qml.expval((1-s)*H0+s*H)    

def BP_DETECT(g,n, bpsteps=False, Fn=1/(9**4)):
    varg=np.var(g)
    tolv=10**(np.floor(np.log10(varg)))
    if tolv<Fn and bpsteps==False:
        print('warning, BP detected')
        print('computed var', varg)
        print('step', n)
        bpsteps=True
    return varg, bpsteps

####VQE stuff
def kandala_VQE(param0, d, Hvqe=Hdef, cost_fc=kandala_cost_fcn, systsz=qubits, max_iterations=mit, conv_tol=ctol,gradDetect=False):
    """
    Function to run VQE. Arguement are the cost function, the initial parameter value

    """
    #
    
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    
    energy=[]
    mingrd=[]
    avggrd=[]
    probs=[]
    nprobs=[]
    var=[]
    thetas=param0
    t0r=time.perf_counter()
    bpsteps=False

    for n in range(max_iterations):
        ##actually runs each optimization step and returns new parameters
        thetas, prev_energy, g= opt.step_and_cost(cost_fc, thetas, H=Hvqe)
        energy.append(kandala_cost_fcn(thetas,Hvqe))
        
        mingrd.append(np.min(g))
        avggrd.append(np.mean(g))
        varg=np.var(g)

        var.append(varg)

        if gradDetect==True:
            
            tolv=10**(np.floor(np.log10(varg)))
            if tolv<1/(9**(8)):
                nprobs.append(n)
                probs.append(varg)
                #print('warning, BP detected')
                #print('computed var', varg)
                #print('step', n)

        conv = np.abs(energy[-1] - prev_energy)
        if conv <= conv_tol:
            break
        
    t1r=time.perf_counter()

    return n, energy[-1], thetas, t1r-t0r, mingrd,avggrd, probs, nprobs, var, energy

def AA_VQE(param0, d, Hvqe=Hdef, H0vqe=H0def, svqe=sdef, cost_fc=cost_fnAA, systsz=qubits, max_iterations=mit, conv_tol=ctol,gradDetect=False):
    
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    
    energy=[]
    svarg=[]
    thetas=param0
    
    bpsteps=False
    t0s=time.perf_counter()
    for n in range(max_iterations):
        ##actually runs each optimization step and returns new parameters
        thetas, prev_energy, g= opt.step_and_cost(cost_fc, thetas, H=Hvqe,H0=H0vqe, s=svqe)
        energy.append(cost_fc(thetas, Hvqe,H0vqe, svqe ))
        

        if gradDetect==True:
            varg, bpsteps=BP_DETECT(g, n, bpsteps)
            svarg.append(varg)

        conv = np.abs(energy[-1] - prev_energy)
        if conv <= conv_tol:
            break
    
    t1s=time.perf_counter()

    return n, energy[-1], thetas, t1s-t0s, svarg, energy

###main loop
for b, bdl in enumerate(bdl_array):
    print('bond length', bdl)

    #Hit,H0it, gsE=MOL_H_BUILD(mol, bdl)
    #Hit, H0it, gsE=ISING_HAM(4, 0.2, 0.5)
    Hit, H0it, gsE=XX_HAM(qubits, bdl)
    Hams.append(Hit)
    GS.append(gsE)
    
    kn, kE, thetas, kt, kgrad, kavggrad, kprobs, knprobs, kvar, kallenergy=kandala_VQE(params0, d, Hvqe=Hit, gradDetect=True)

    

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(np.linspace(0, kn, kn+1), kallenergy, label='b='+str(b))
    # ax2.plot(knprobs, kprobs, 'b.', label='problem points')
    # ax1.plot(np.linspace(0, kn, kn+1) ,kavggrad,label='avg grad'  )
    # ax1.plot(np.linspace(0, kn, kn+1) ,kgrad, label='min grad' )
    # ax2.plot(np.linspace(0, kn, kn+1) ,kvar, label='var' )
    # plt.legend()
    # plt.show()
    print('HEA solution', kn, kE, kt)
    print('actual Ground state energy', gsE)
    kits.append(kn)
    kenergy.append(kE)
    ktimes.append(kt)
    sEplotlist=[]
    params0=np.random.rand(3*d*qubits+2*qubits)
    
###comment out_ Strg #
    sntot=0
    for sind, sit in enumerate(sarray):    
        n, E,thetas, ts, svarg, senergies=AA_VQE(params0, d, Hvqe=Hit, H0vqe=H0it, svqe=sit, gradDetect=False )

        senergy.append(E)
        sangle.append(thetas)
        sn.append(n)
        st.append(ts)
        sdictvarg.update({"sit_is_"+str(sit): svarg}) 
        print("it solution", n)
        params0=thetas
        sntot=sntot+n+1
        sEplotlist=sEplotlist+senergies
        
    
    print('AAVQE solution', senergy[-1])
    # print('sntot is', sntot)
    # print('saved energies is', len(sEplotlist))
    ax2.plot(np.linspace(0, sntot, sntot), np.array(sEplotlist))

plt.legend()
plt.show()
###save stuff
data={'GSE': GS,'ssteps':ssteps,'sits': sn, 'senergy':senergy, 'sangles':sangle,'stimes':st,'s_vars':sdictvarg ,'kits':kits, 'kenergy':kenergy, 'kangle':kangles, 'ktimes': ktimes, 'k_vars': kvarg, 'number_interd': numpoints, 'interatom_d': bdl_array, 'init_kparam': params0, 'Hams': Hams, 'ansatz_depth': d, 'solver':'GD_0.04' , 'max_iterations': mit, 'conv_tol': ctol}

if ifsave==True:
    filename='kandala_'+mol+'_'+str(numpoints)+'_iads.pkl'
    script_path = os.path.abspath(__file__)
    save_path=script_path.replace("01_code\AAVQE_Kandala_ansatz.py", "03_data")
    completename = os.path.join(save_path, filename) 
    
    with open(completename,'wb') as file:
        pickle.dump(data, file)