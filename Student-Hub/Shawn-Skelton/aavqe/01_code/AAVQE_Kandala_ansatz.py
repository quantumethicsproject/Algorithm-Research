# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 21:05:48 2023

@author: skelt
"""
import pennylane as qml
from pennylane import numpy as np
#from pennylane import qchem
import time
#import matplotlib.pyplot as plt
import pickle
import os.path

from qiskit.providers.fake_provider import *
import matplotlib.pyplot as plt
# from pennylane_cirq import ops as cirq_ops

from tqdm import tqdm

import itertools

# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# from qiskit.visualization import plot_histogram
# from qiskit_aer.noise import NoiseModel
#from qiskit.providers.fake_provider.backends.manila import FakeManila

'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{braket}')

####constants
ifsave=True
#mol='H2'
numpoints=6
d=1
ctol=1.6*10**(-3)
mit=200
ssteps=20
p=0.05

###want to automate eventually:
qubits=3
HNAME='1XX3'
NMODEL="FakeManila" #"bitflippenny=0.05"#"depolcirq=0.05"
###stuff for the variance: want an randomized order of magnitude bound for the variance

### FILE PATHS
script_path = os.path.abspath(__file__)
# save_path=script_path.replace("01_code\AAVQE_Kandala_ansatz.py", "03_data")
save_path = "aavqe/03_data"

###JEFF'S NOISE MODEL CODE###
def configured_backend():
    # backend = provider.get_backend("ibm_osaka") # uncomment this line to use a real IBM device
    backend = FakeManila()
    
    # backend.options.update_options(...)
    return backend

# create our devices to run our circuits on
dev=qml.device('default.qubit', wires=qubits)
noise_strength = p
dev_N = qml.device("default.mixed", wires=qubits)

if NMODEL == "FakeManila":
    dev_N = qml.device("qiskit.remote", wires=qubits, backend=configured_backend()) # device for real IBM devices noisy simulators
if "cirq" in NMODEL:
    dev_N=qml.device("cirq.mixedsimulator", wires=qubits)

###the shape we want is tensor([...], requires_grad=true)
params0all=np.random.rand(3*d*qubits+2*qubits) #np.array([theta0]*(3*d*qubits+2*qubits))

GS=[]
kits=[]
Nkenergy=[]
Nkits=[]

sarray=np.linspace(0, 1, ssteps) 

# available_data = qml.data.list_datasets()["qchem"][mol]["STO-3G"]
# bdl_array=available_data[1:2]

# bdl_array=np.linspace(-1, 1, numpoints)
bdl_array=np.array([qubits])

def MOL_H_BUILD(mol, bdl):
    part = qml.data.load("qchem", molname=mol, basis="STO-3G", bondlength=bdl, attributes=["molecule", "hamiltonian", "fci_energy"])[0]
    H=part.hamiltonian
    H0=qml.Hamiltonian(np.ones(qubits)/2, [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2), qml.PauliX(3)])
    #FCI:  Full configuration interaction (FCI) energy computed classically should be in Hatrees
    gsE=part.fci_energy
    return H, H0, gsE

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

def SUBSET_GENERATOR(sites):
    if sites==4:
        combinations_object=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    elif sites==5:
        combinations_object=[[0, 1, 2], [0, 1, 3],[0, 1, 4], [0, 2, 3],[0, 2, 4], [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    elif sites==6:
        combinations_object=[[0, 1, 2], [0, 1, 3],[0, 1, 4], [0, 1, 5], [0, 2, 3],[0, 2, 4],[0, 2, 5], [0, 3, 4],[0, 3, 5],[0, 4, 5], [1, 2, 3], [1, 2, 4],[1, 2, 5], [1, 3, 4],[1, 3, 5], [1, 4, 5],[2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]
        print(len(combinations_object))
    elif sites==7:
        combinations_object=[[0, 1, 2], [0, 1, 3],[0, 1, 4], [0, 1, 5],[0, 1, 6], [0, 2, 3],[0, 2, 4],[0, 2, 5],[0, 2, 6], [0, 3, 4],[0, 3, 5], [0, 3, 6], [0, 4, 5], [0, 4, 6], [0, 5, 6], [1, 2, 3], [1, 2, 4],[1, 2, 5], [1, 2, 6],[1, 3, 4],[1, 3, 5],[1, 3, 6], [1, 4, 5],[1, 4, 6],[2, 3, 4], [2, 3, 5], [2, 3, 6],[2, 4, 5], [2, 5, 6], [3, 4, 5], [3, 4, 6],[3, 5, 6], [4, 5, 6]]
        
    return combinations_object


def EXACT_COVER_HAM(sites):
    subsets=SUBSET_GENERATOR(sites)#[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    
    coeffs=[1, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    fullcoeffs=[]
    fullGset=[]

    for s, subset in enumerate(subsets):
        Gset=[qml.Identity(wires=subset[0])@qml.Identity(wires=subset[1])@qml.Identity(wires=subset[2])] + [qml.PauliZ(i) for i in subset] + [qml.PauliZ(subset[0])@qml.PauliZ(subset[1]), qml.PauliZ(subset[0])@qml.PauliZ(subset[2]), qml.PauliZ(subset[1])@qml.PauliZ(subset[2]) ]
        #Gset=[qml.Identity(wires=subset)] + [qml.PauliZ(i) for i in subset]+ [qml.PauliZ(subset[0])@qml.PauliZ(subset[1]), qml.PauliZ(subset[0])@qml.PauliZ(subset[2]), qml.PauliZ(subset[1])@qml.PauliZ(subset[2]) ]
        
        fullcoeffs=fullcoeffs+coeffs
        fullGset=fullGset+Gset
        
    
    H0=qml.Hamiltonian(np.ones(sites)/(sites), [qml.PauliX(i) for i in range(0, sites)])
    H=qml.Hamiltonian(fullcoeffs, fullGset)
    
    
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
    #qml.BasisState(np.zeros(len(wires)), wires=wires)
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

@qml.qnode(dev_N, interface="autograd")
def kandala_cost_fcn_noise(param, H=Hdef):
    kandala_circuit(param, range(qubits), d)
    # if "cirq" in NMODEL:
        # if NMODEL=="bitflipcirq=0.05":
        #     [cirq_ops.BitFlip(p, wires=bit) for bit in range(qubits)]
        # elif NMODEL=="depolcirq=0.05":
        #     [cirq_ops.Depolarize(p, wires=bit) for bit in range(qubits)]
    if NMODEL=="bitflippenny=0.05":
        [qml.BitFlip(p, wires=i) for i in range(qubits)]
    elif NMODEL=="FakeManila":
        return qml.expval(H)
    else:
        print('warning, noise model not recognized')
    return qml.expval(H)

@qml.qnode(dev, interface="autograd")
def cost_fn(param, H=Hdef):
    circuit(param, wires=range(qubits))
    return qml.expval(H)

@qml.qnode(dev, interface="autograd")
def cost_fnAA(param, H=Hdef, H0=H0def, s=sdef): 
    kandala_circuit(param, range(qubits), d)
    return qml.expval((1-s)*H0+s*H)    

@qml.qnode(dev_N, interface="autograd")
def cost_fnAA_noise(param, H=Hdef, H0=H0def, s=sdef): 
    kandala_circuit(param, range(qubits), d)
    # if "cirq" in NMODEL:
    #     if NMODEL=="bitflipcirq=0.05":
    #         [cirq_ops.BitFlip(p, wires=bit) for bit in range(qubits)]
    #     elif NMODEL=="depolcirq=0.05":
    #         [cirq_ops.Depolarize(p, wires=bit) for bit in range(qubits)]
    #     # for bit in range(qubits):
    #     #     #cirq_ops.Depolarize(p, wires=bit)
    #     #     cirq_ops.BitFlip(p, wires=bit)
    # elif NMODEL=="bitflippenny=0.05":
    #     [qml.BitFlip(p, wires=i) for i in range(qubits)]
    if NMODEL=="FakeManila":
        return qml.expval(H)
    else:
        print('warning, noise model not recognized')

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

    for n in tqdm(range(max_iterations)):
        ##actually runs each optimization step and returns new parameters
        thetas, prev_energy= opt.step_and_cost(cost_fc, thetas, H=Hvqe)
        energy.append(kandala_cost_fcn(thetas,Hvqe))
        
        # mingrd.append(np.min(g))
        # avggrd.append(np.mean(g))
        # varg=np.var(g)

        # var.append(varg)

        # if gradDetect==True:
            
        #     tolv=10**(np.floor(np.log10(varg)))
        #     if tolv<1/(9**(8)):
        #         nprobs.append(n)
        #         probs.append(varg)
        #         #print('warning, BP detected')
        #         #print('computed var', varg)
        #         #print('step', n)
        # 'mingrad': mingrd ,'avggrd': avggrd, 'probs': probs, 'nprobs': nprobs,'vars': varg,
        conv = np.abs(energy[-1] - prev_energy)
        if conv <= conv_tol:
            break
        
    t1r=time.perf_counter()
    DATA={'its':n, 'gsEest':energy[-1], 'angles':thetas, 'timer': t1r-t0r,'energies':energy,}
    return DATA

def AA_VQE(param0, d, Hvqe=Hdef, H0vqe=H0def, svqe=sdef, cost_fc=cost_fnAA, systsz=qubits, max_iterations=mit, conv_tol=ctol,gradDetect=False):
    
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    
    energy=[]
    svarg=[]
    thetas=param0
    
    bpsteps=False
    t0s=time.perf_counter()
    for n in range(max_iterations):
        ##actually runs each optimization step and returns new parameters
        thetas, prev_energy= opt.step_and_cost(cost_fc, thetas, H=Hvqe,H0=H0vqe, s=svqe)
        energy.append(cost_fc(thetas, Hvqe,H0vqe, svqe ))
        

        # if gradDetect==True:
        #     varg, bpsteps=BP_DETECT(g, n, bpsteps)
        #     svarg.append(varg)

        conv = np.abs(energy[-1] - prev_energy)
        if conv <= conv_tol:
            break
    
    t1s=time.perf_counter()
    SDATA={'its': n, 'energy':energy[-1], 'sEnergy_all': energy, 'angles':thetas,'stimes':t1s-t0s,'vars': svarg, }   
    return SDATA

def RUN_AA_VQE(sarray, initparams,d, Hit, H0it, cost_fc=cost_fnAA):
        sntot=0
        SDATA={}
        params=initparams

        sEplotlist=[]
        senergy =[]
        sn=[]
        st=[]

        for sind, sit in tqdm(enumerate(sarray)):
            svarg="sit_is_"+str(sit)
            SinstDATA=AA_VQE(params, d, Hvqe=Hit, H0vqe=H0it, svqe=sit, gradDetect=False )
            SDATA['svarg']=SinstDATA
            params=SinstDATA['angles']
            senergy.append(SinstDATA['energy'])
            sn.append(SinstDATA['its'])
            st.append(SinstDATA['stimes'])
            sntot=sntot+sn[-1]+1
            sEplotlist=sEplotlist+SinstDATA['sEnergy_all']
            
        SDATA['fulln']=sntot
        SDATA['fullgsE']=senergy
        SDATA['fullenergy']=sEplotlist
        SDATA['fulltimes']=st
        return SDATA, sEplotlist

##main loop
data={'ssteps':ssteps, 'noisetype':NMODEL, 'noiseparam':p ,'interatom_d': bdl_array, 'init_kparam': params0all,  'ansatz_depth': d, 'solver':'GD_0.04' , 'max_iterations': mit, 'conv_tol': ctol}

for b, bdl in enumerate(bdl_array):
    print('bond length', bdl)
    #Hit, H0it, gsE=XX_HAM(qubits, bdl)
    Hit, H0it,gsE=EXACT_COVER_HAM(bdl)
    GS.append(gsE)
    bdictname='b_'+str(np.around(bdl))+'_data'
    params=params0all
    #KDATA=kandala_VQE(params, d, Hvqe=Hit, gradDetect=True, max_iterations=mit*ssteps)
    #kallenergy=KDATA['energies']

    params=params0all
    NKDATA=kandala_VQE(params, d, Hvqe=Hit, cost_fc=kandala_cost_fcn_noise, max_iterations=mit*ssteps)
    Nkits.append(NKDATA['its'])
    Nkenergy.append(NKDATA['gsEest'])
    Nkallenergy=NKDATA['energies']
    print('noisy done', Nkits[-1])
    
    ###make a figure with some subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    
    ax1.plot(np.linspace(0, Nkits[-1], Nkits[-1]+1), Nkallenergy, c='r', marker=1, label='Noisy HEA VQE')
    ax1.axhline(y=gsE,xmin=0,xmax=3,c="blue",linewidth=0.5,zorder=0, label="Analytic GSE")

    ax2.plot(np.linspace(0, Nkits[-1], Nkits[-1]+1), Nkallenergy, c='r', marker=1, label='Noisy HEA VQE')
    #ax2.plot(np.linspace(0, KDATA['its'], KDATA['its']+1), kallenergy, c='blue', marker=1, label='HEA VQE')
    ax2.axhline(y=gsE,xmin=0,xmax=3,c="blue",linewidth=0.5,zorder=0, label="Analytic GSE")

    ax1.set_title('VQE E vs iteration')
    ax1.set_ylabel(r'$\braket{U^{\dag}(\theta)e^{iH}U(\theta)}$')
    ax1.set_xlabel(r'iteration $n$')

    SDATA, sEplotlist=RUN_AA_VQE(sarray, params0all, d, Hit, H0it, )
    NSDATA, NsEplotlist=RUN_AA_VQE(sarray, params, d, Hit, H0it,  cost_fc=cost_fnAA_noise )
    print('noisy AAVQE done')
    #NSDATA=[]
    #NKDATA=[]
    ax1.plot(np.linspace(0, NSDATA['fulln'], NSDATA['fulln']), np.array(NsEplotlist), c='blue', marker=3,label='Noisy AAVQE' )
    filename='AAVQE_HEA_'+HNAME+'_lambda='+str(np.around(bdl, 2))+NMODEL+'.png'

    ax3.plot(np.linspace(0, SDATA['fulln'], SDATA['fulln']), np.array(sEplotlist), c='r', marker=1, label='AAVQE')
    ax3.plot(np.linspace(0, NSDATA['fulln'], NSDATA['fulln']), np.array(NsEplotlist), c='blue', marker=2, label='Noisy AAVQE')
    
    ax3.axhline(y=gsE,xmin=0,xmax=3,c="blue",linewidth=0.5,zorder=0, label="Analytic GSE")

    # script_path = os.path.abspath(__file__)
    # save_path=script_path.replace("01_code\AAVQE_Kandala_ansatz.py", "03_data")
    completefigname = os.path.join(save_path, filename) 
#    bdict={'bdl':bdl, 'gsE': gsE, 'hamiltonian': Hit, 'sdata': SDATA,'Nsdata': NSDATA, 'kdata': KDATA, 'Nkdata': NKDATA}
    bdict={'bdl':bdl, 'gsE': gsE, 'hamiltonian': Hit, 'sdata': SDATA,'Nsdata': NSDATA,  'Nkdata': NKDATA}

    data[bdictname]=bdict
    if ifsave==True:
        plt.savefig(completefigname)
        filename='AAVQE_w_'+NMODEL+'_'+HNAME+'_lambda_'+str(np.around(bdl, 2))+'.pkl'
        completename=os.path.join(save_path, filename) 
        with open(completename,'wb') as file:
            pickle.dump(bdict, file)
    

plt.legend()
###save stuff

if ifsave==True:
    filename='AAVQE_w_'+NMODEL+'_'+HNAME+'_'+str(numpoints)+'_iads.pkl'
    completename = os.path.join(save_path, filename) 
    with open(completename,'wb') as file:
        pickle.dump(data, file)
