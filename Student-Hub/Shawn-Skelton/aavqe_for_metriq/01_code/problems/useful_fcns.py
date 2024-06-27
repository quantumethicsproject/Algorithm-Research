from pennylane import numpy as np
import pennylane as qml
from pennylane import qchem

def MOL_H_BUILD(mol, bdl):
    """"
    grabs a Hamiltonian from pennylane's library 
    mol is the pennylane string which refers to the desired molecule and bdl is the bondlength (needs to be one of pennylane's options)
    """
    part = qml.data.load("qchem", molname=mol, basis="STO-3G", bondlength=bdl, attributes=["molecule", "hamiltonian", "fci_energy"])[0]
    H=part.hamiltonian
    H0=qml.Hamiltonian(np.ones(qubits)/2, [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2), qml.PauliX(3)])
    #FCI:  Full configuration interaction (FCI) energy computed classically should be in Hatrees
    gsE=part.fci_energy
    return H, H0, gsE

def EASY_HAM(q):
    """
    builds the 'easy' hamiltonian. May want to play around with this choice
    need to rebuild for any number of qubits
    """
    H0=qml.Hamiltonian(np.ones(q)/2, [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2), qml.PauliX(3)])
    return H0

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
    """
    Generates all of the possible site combinations for the exact cover problem, from 3 to 7 qubits
    sites: number of qubits, ie sites, ie problem variables
    """
    if sites==3:
        combinations_object=[[0, 1, 2]]
    elif sites==4:
        combinations_object=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    elif sites==5:
        combinations_object=[[0, 1, 2], [0, 1, 3],[0, 1, 4], [0, 2, 3],[0, 2, 4], [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    elif sites==6:
        combinations_object=[[0, 1, 2], [0, 1, 3],[0, 1, 4], [0, 1, 5], [0, 2, 3],[0, 2, 4],[0, 2, 5], [0, 3, 4],[0, 3, 5],[0, 4, 5], [1, 2, 3], [1, 2, 4],[1, 2, 5], [1, 3, 4],[1, 3, 5], [1, 4, 5],[2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]
        print(len(combinations_object))
    elif sites==7:
        combinations_object=[[0, 1, 2], [0, 1, 3],[0, 1, 4], [0, 1, 5],[0, 1, 6], [0, 2, 3],[0, 2, 4],[0, 2, 5],[0, 2, 6], [0, 3, 4],[0, 3, 5], [0, 3, 6], [0, 4, 5], [0, 4, 6], [0, 5, 6], [1, 2, 3], [1, 2, 4],[1, 2, 5], [1, 2, 6],[1, 3, 4],[1, 3, 5],[1, 3, 6], [1, 4, 5],[1, 4, 6],[2, 3, 4], [2, 3, 5], [2, 3, 6],[2, 4, 5], [2, 5, 6], [3, 4, 5], [3, 4, 6],[3, 5, 6], [4, 5, 6]]
        
    return combinations_object



def EC_HAM(sites):
    """
    Builds a hamiltonian for the exact cover problem and computes the ground state energy using exact diagonalization
    In math, the $n$ site(qubit) Ising hamiltonian is $H=J.....\sum_{i=0}^{n-1} X_iX_{i+1}+\lambda\sum_{i=0}^nX_i$
    H_0 is the simple hamiltonian we need for the AAVQE step
    sites: number of qubits ie sites ie classical variables
    """
    subsets=SUBSET_GENERATOR(sites)
    
    coeffs=[1, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    fullcoeffs=[]
    fullGset=[]

    for s, subset in enumerate(subsets):
        Gset=[qml.Identity(wires=subset[0])@qml.Identity(wires=subset[1])@qml.Identity(wires=subset[2])] + [qml.PauliZ(i) for i in subset] + [qml.PauliZ(subset[0])@qml.PauliZ(subset[1]), qml.PauliZ(subset[0])@qml.PauliZ(subset[2]), qml.PauliZ(subset[1])@qml.PauliZ(subset[2]) ]
        fullcoeffs=fullcoeffs+coeffs
        fullGset=fullGset+Gset
        
    
    H0=qml.Hamiltonian(np.ones(sites)/(sites), [qml.PauliX(i) for i in range(0, sites)])
    H=qml.Hamiltonian(fullcoeffs, fullGset)
    
    
    eigs=np.linalg.eigvals(qml.matrix(H))
    gse=min(np.real(eigs))
    
    return H, H0, gse
