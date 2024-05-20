import pennylane as qml
from pennylane import numpy as np

####DEFINE ALL THE COST FUNCTIONS WRT DIFFERENT PARAMETERIZED CIRCUITS###########
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
            
            
def kandala_cost_fcn(H, param, d=1):
    kandala_circuit(param, range(qubits), d)
    return qml.expval(H)

    
def cost_fn(param, instance):
    circuit(param, wires=range(instance.qubits))
    H=instance.H
    return qml.expval(H)

   
def cost_fnAA(param,H, H0,  qubits, s=1, d=1): 
    kandala_circuit(param, range(qubits), d)
    return qml.expval((1-s)*H0+s*H) 

