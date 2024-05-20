from pennylane import numpy as np
import pennylane as qml

def MOL_H_BUILD(mol, bdl):
    """"
    grabs a Hamiltonian from pennylane's library 
    """
    # print(qml.data.list_attributes("qchem"))
    part = qml.data.load("qchem", molname=mol, basis="STO-3G", bondlength=bdl, attributes=["molecule", "hamiltonian", "hf_state", "fci_energy"])[0]
    H=part.hamiltonian

    #FCI:  Full configuration interaction (FCI) energy computed classically should be in Hatrees
    gsE=part.fci_energy
    qubits=len(part.hf_state)

    return H, gsE, qubits

def EASY_HAM(q):
    """
    builds the 'easy' hamiltonian. May want to play around with this choice
    need to rebuild for any number of qubits
    """
    H0=qml.Hamiltonian(np.ones(q)/2, [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2), qml.PauliX(3)])
    return H0

