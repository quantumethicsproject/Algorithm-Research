# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 21:05:48 2023

@author: skelt
"""
import pennylane as qml
from pennylane import numpy as np
import time
import pickle
import os.path
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

import problems.useful_fcns as ufs
'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{braket}')

####CONSTANTS WHICH THE USER SETS FOR EACH RUN
ifsave=True
run_vqe=True
qubits=13
HNAME='4XX13'
print('hamiltonian is', HNAME)

NMODEL='nonoise'#'bitflippenny=0.05' #"FakeManila"#"bitflippenny=0.05" #"bitflippenny=0.05"#"depolcirq=0.05"
device='notsess' #'sess'
numpoints=8
bdl_array=np.linspace(-1, 1, numpoints)
#bdl_array=np.array([qubits])
# available_data = qml.data.list_datasets()["qchem"][mol]["STO-3G"]
# bdl_array=available_data[1:2]
sdef=1
#mol='H2'

###CONSTANTS WHICH SHOULD STAY CONSISTENT
d=1
ctol=1.6*10**(-3)
mit=200
ssteps=20
sarray=np.linspace(0, 1, ssteps) 
p=0.05
Hdef,H0def, gsEdef=ufs.XX_HAM(qubits, bdl_array[0])

### FILE PATHS
script_path = os.path.abspath(__file__)
if device=='sess':
    save_path = "aavqe\\03_data"
    save_path=script_path.replace("01_code\\AAVQE_HEA_SIM.py", "03_data")
else:
    script_path = os.path.abspath(__file__)
    #save_path="aavqe/03_data"
    save_path=script_path.replace("01_code/AAVQE_HEA_SIM.py", "03_data")
                                 
###JEFF'S NOISE MODEL CODE###
def configured_backend():
    # backend = provider.get_backend("ibm_osaka") # uncomment this line to use a real IBM device
    backend = stupid.FakeManila()
    # backend.options.update_options(...)
    return backend

###CREATE DEVICES 
dev=qml.device('default.qubit', wires=qubits)
noise_strength = p
dev_N = qml.device("default.mixed", wires=qubits)

if NMODEL == "FakeManila":
    dev_N = qml.device("qiskit.remote", wires=qubits, backend=configured_backend()) # device for real IBM devices noisy simulators
if "cirq" in NMODEL:
    from pennylane_cirq import ops as cirq_op
    dev_N=qml.device("cirq.mixedsimulator", wires=qubits)

###RANDOMIZE THE INITIAL PARAMETERS
params0all=np.random.rand(3*d*qubits+2*qubits) #np.array([theta0]*(3*d*qubits+2*qubits))

###DEFINE ARRAYS FOR IMPORTANT DATA
GS=[]
kits=[]
Nkenergy=[]
Nkits=[]

####CIRCUITS FOR THE HEA
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
    #     
def HEA_circuit(param, wires, d):
    ###simplified circuit from http://arxiv.org/abs/1704.05018
    ###indexing looks a bit messy its a 1d list 
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
            qml.RZ(param[indtrack], wires=[q])
            qml.RX(param[indtrack+1], wires=[q])
            qml.RZ(param[indtrack+2], wires=[q])
            indtrack=indtrack+3
        
def q4circuit(param, wires):
    """
    A very simple VQA circuit which only works for the 4-qubit H2 case.
    param is a float
    wires is an 1x 4 array
    """
    hf = qml.qchem.hf_state(electrons, qubits)
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

def scircuit(param, wires):
    """
    meant to compare AAVQE and VQE for the H2 mol, where a 1param circuit can succeed: 
    param is a float
    wires is an 1x 4 array
    """
    qml.BasisState(np.zeros(len(wires)), wires=wires)
    
####Kandala cost function
@qml.qnode(dev, interface="autograd")
def HEA_cost_fcn(param, H=Hdef):
    """
    runs the HEA and then measures operator e^H
    param: an numpy array with tensor elements
    H: the hamiltonian required
    """
    HEA_circuit(param, range(qubits), d)
    return qml.expval(H)

@qml.qnode(dev_N, interface="autograd")
def HEA_cost_fcn_noise(param, H=Hdef):
    """
    runs the HEA, simulates noise, and then measures operator e^H
    param: an numpy array with tensor elements
    H: the hamiltonian required
    """
    HEA_circuit(param, range(qubits), d)
    
    if NMODEL=="bitflippenny=0.05":
        [qml.BitFlip(p, wires=i) for i in range(qubits)]
    elif NMODEL=="FakeManila":
        return qml.expval(H)
    else:
        print('warning, noise model not recognized')
    return qml.expval(H)

@qml.qnode(dev, interface="autograd")
def cost_fn(param, H=Hdef):
    """
    runs the simple ansatz from the 4-qubit case and then measures operator e^H
    param: an numpy array with tensor elements
    H: the hamiltonian required
    """
    q4circuit(param, wires=range(qubits))
    return qml.expval(H)

@qml.qnode(dev, interface="autograd")
def cost_fnAA(param, H=Hdef, H0=H0def, s=sdef):
    """
    runs the HEA and then measures operator e^H
    param: an numpy array with tensor elements
    H: the hamiltonian required
    """ 
    HEA_circuit(param, range(qubits), d)
    #print('type of s', type(float(s)))
    return qml.expval(qml.simplify(float(1-s)*H0+float(s)*H))    
    #return qml.expval(0.3*H0)

@qml.qnode(dev_N, interface="autograd")
def cost_fnAA_noise(param, H=Hdef, H0=H0def, s=sdef): 
    """
    runs the HEA, simulates noise, and then measures operator e^(H_aavqe)
    param: an numpy array with tensor elements
    H: the hamiltonian required
    H0: the simple hamiltonian
    s: a float, the step of the 'adiobatic inspired' iteration
    """
    HEA_circuit(param, range(qubits), d)
    
    if NMODEL=="bitflippenny=0.05":
        [qml.BitFlip(p, wires=i) for i in range(qubits)]
    elif NMODEL=="FakeManila":
        return qml.expval((1-s)*H0+s*H)
    else:
        print('warning, noise model not recognized')
    
    #return qml.expval(H0)
    return qml.expval(float(1-s)*H0+float(s)*H)

####VQE SOLVERS
def kandala_VQE(param0, d, Hvqe=Hdef, cost_fc=HEA_cost_fcn, systsz=qubits, max_iterations=mit, conv_tol=ctol):
    """
    Function to run standard VQE
    param0: the initial parameter list
    d: the layers for the HEA
    Hvqe: the hamiltonian
    cost_fc: the cost function, default is the HEA
    systsz: system size, the number of qubits
    max_iterations: the maximum allowed number of iterations
    conv_tol: the convergence bound
    returns a python dictionary with the number of iterations, VQA solution, the solution parameters, the solution time, and all of the intermediate energy solutions
    """
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    energy=[]
    thetas=param0

    t0r=time.perf_counter()
    bpsteps=False
    ##actually runs each optimization step and returns new parameters
    for n in tqdm(range(max_iterations)):
        thetas, prev_energy= opt.step_and_cost(cost_fc, thetas, H=Hvqe)
        energy.append(HEA_cost_fcn(thetas,Hvqe))

        conv = np.abs(energy[-1] - prev_energy)
        if conv <= conv_tol:
            break
        
    t1r=time.perf_counter()
    DATA={'its':n+1, 'gsEest':energy[-1], 'angles':thetas, 'timer': t1r-t0r,'energies':energy}
    return DATA

def AA_VQE(param0, d, Hvqe=Hdef, H0vqe=H0def, svqe=sdef, cost_fc=cost_fnAA, systsz=qubits, max_iterations=mit, conv_tol=ctol):
    """
    Function to run a singular AAVQE step
    param0: the initial parameter list
    d: the layers for the HEA
    Hvqe: the hamiltonian
    H0vqe: the 'easy' hamiltonian
    svqe: the step along approximation path
    cost_fc: the cost function, default is the HEA modified for AAVQE
    systsz: system size, the number of qubits
    max_iterations: the maximum allowed number of iterations
    conv_tol: the convergence bound
    returns a python dictionary with the number of iterations, VQA solution, the solution parameters, the solution time, and all of the intermediate energy solutions
    """
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    
    energy=[]
    thetas=param0
    
    bpsteps=False
    t0s=time.perf_counter()

    ##actually runs each optimization step and returns new parameters
    for n in range(max_iterations):
        thetas, prev_energy= opt.step_and_cost(cost_fc, thetas, H=Hvqe,H0=H0vqe, s=svqe)
        energy.append(cost_fc(thetas, Hvqe,H0vqe, svqe ))
        
        conv = np.abs(energy[-1] - prev_energy)
        if conv <= conv_tol:
            break
    
    t1s=time.perf_counter()
    SDATA={'its': n+1, 'gsEest':energy[-1],  'angles':thetas, 'timer':t1s-t0s, 'energies': energy,}   
    return SDATA

def RUN_AA_VQE(sarray, param0,d, Hvqe, H0vqe, cost_fc=cost_fnAA, systsz=qubits, max_iterations=mit, conv_tol=ctol):
    """
    Function to run every AAVQE step
    sarray: the array of all approximation steps from s=0 to s=1
    param0: the initial parameter list
    d: the layers for the HEA
    Hvqe: the hamiltonian
    H0vqe: the 'easy' hamiltonian
    cost_fc: the cost function, default is the HEA modified for AAVQE
    systsz: system size, the number of qubits
    max_iterations: the maximum allowed number of iterations
    conv_tol: the convergence bound
    returns a python dictionary with the number of iterations, VQA solution, the solution parameters, the solution time, and all of the intermediate energy solutions
    """

    ###DEFINE VARIABLES TO TRACK n, t, E for each soltuion, and to store data for each
    sntot=0
    SDATA={}
    params=param0

    sEplotlist=[]
    senergy =[]
    sn=[]
    st=[]
    ###RUN EACH AAVQE STEP AND SAVE DATA
    for sind, sit in tqdm(enumerate(sarray)):
        SinstDATA=AA_VQE(params, d, Hvqe, H0vqe, sit, cost_fc, systsz, max_iterations, conv_tol )
        SDATA["sit_is_"+str(np.around(sit, 3))]=SinstDATA
        params=SinstDATA['angles']
        senergy.append(SinstDATA['gsEest'])
        sn.append(SinstDATA['its'])
        st.append(SinstDATA['timer'])
        sntot=sntot+SinstDATA['its']
        sEplotlist=sEplotlist+SinstDATA['energies']
            
    SDATA['fulln']=sntot
    SDATA['fullgsE']=senergy
    SDATA['fullenergy']=sEplotlist
    SDATA['fulltimes']=st
    return SDATA, sEplotlist

##main loop
data={'ssteps':ssteps, 'noisetype':NMODEL, 'noiseparam':p ,'interatom_d': bdl_array, 'init_kparam': params0all,  'ansatz_depth': d, 'solver':'GD_0.04' , 'max_iterations': mit, 'conv_tol': ctol}
for b, bdl in enumerate(bdl_array):
    print('bond length', bdl)
    
    if "XX" in HNAME:
        filename='AAVQE_w_'+NMODEL+'_'+HNAME+'_'+str(bdl)+'_instance'+'.pkl'
        Hit, H0it, gsE=ufs.XX_HAM(qubits, bdl)
    elif "EC" in HNAME:
        filename='AAVQE_w_'+NMODEL+'_'+HNAME+'_sites_'+str(qubits)+'.pkl'
        Hit, H0it, gsE=ufs.EC_HAM(qubits)
    
    GS.append(gsE)
    bdictname='b_'+str(np.around(bdl))+'_data'

    if run_vqe==True:
        KDATA=kandala_VQE(params0all, d, Hvqe=Hit,  max_iterations=mit*ssteps)
        kallenergy=KDATA['energies']
        
    SDATA, sEplotlist=RUN_AA_VQE(sarray, params0all, d, Hit, H0it, )
    
    if NMODEL!='nonoise':
        NKDATA=kandala_VQE(params0all, d, Hvqe=Hit, cost_fc=HEA_cost_fcn_noise, max_iterations=mit*ssteps)
        Nkits.append(NKDATA['its'])
        Nkenergy.append(NKDATA['gsEest'])
        Nkallenergy=NKDATA['energies']
        print('noisy VQE done', Nkits[-1])
        NSDATA, NsEplotlist=RUN_AA_VQE(sarray, params0all, d, Hit, H0it,  cost_fc=cost_fnAA_noise )
        ###SAVE INSTANCE DATA
        if run_vqe==True:
            bdict={'bdl':bdl, 'gsE': gsE, 'hamiltonian': Hit, 'sdata': SDATA,'Nsdata': NSDATA, 'kdata': KDATA, 'Nkdata': NKDATA}
        else:
            bdict={'bdl':bdl, 'gsE': gsE, 'hamiltonian': Hit, 'sdata': SDATA,'Nsdata': NSDATA,  'Nkdata': NKDATA}
    
        print('noisy AAVQE done', NSDATA['fulln'])
    else:
        ###SAVE INSTANCE DATA
        if run_vqe==True:
            bdict={'bdl':bdl, 'gsE': gsE, 'hamiltonian': Hit, 'sdata': SDATA,'kdata': KDATA}
        else:
            bdict={'bdl':bdl, 'gsE': gsE, 'hamiltonian': Hit, 'sdata': SDATA,}
    
    data[bdictname]=bdict

    completename = os.path.join(save_path, filename) 
    if ifsave==True:
        with open(completename,'wb') as file:
            pickle.dump(bdict, file)

###SAVE EVERYTHING IN ONE FILE; MORE CONVENIANT USUALLY
if ifsave==True:
    filename='AAVQE_w_'+NMODEL+'_'+HNAME+'_'+str(numpoints)+'_iads.pkl'
    completename = os.path.join(save_path, filename) 
    with open(completename,'wb') as file:
        pickle.dump(data, file)
os.system('afplay /System/Library/Sounds/Sosumi.aiff')
