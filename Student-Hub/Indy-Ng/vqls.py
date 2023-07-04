"""Helper function code"""

from typing import (List, Tuple)

import functools as ft

import pennylane as qml
from pennylane import numpy as np
from pennylane import Identity, PauliX, PauliY, PauliZ


pauli_dict = {"I": Identity.compute_matrix(), "X": PauliX.compute_matrix(), "Y": PauliY.compute_matrix(), "Z": PauliZ.compute_matrix()}


def A_to_num (n_qubits: int, coefs: np.tensor, terms: List[str]):
    
    if len(coefs) != len(terms):
        raise ValueError("Number of coefficients does not match number of terms.")
    
    if n_qubits <= 0:
        raise ValueError("Number of qubits is not a number greater than 0.")
    
    terms_len = len(terms)
    for i in range(terms_len):
        if len(terms[i]) != n_qubits:
            raise ValueError("Number of terms in each Pauli gate combination must be the same as number of qubits.")
        

    dim = 2**n_qubits
    mat = np.zeros((dim, dim), dtype=np.complex64)

    for (c, pauli) in zip(coefs, terms):
        pauli = [pauli_dict[key] for key in pauli]
        if pauli == ["I"]:
            mat += c * ft.reduce(np.kron, pauli)
        else:
            mat += c * ft.reduce(np.kron, pauli)
        
    return mat

def generate_weights(n_qubits:int, layers:int, q_delta:float):
    if n_qubits <= 0:
        raise ValueError("Number of qubits is not a number greater than 0.")
    
    if layers < 0:
        raise ValueError("Number of layers is not a number greater than or equal to 0.")
    
    dimension = 0

    # For fixed layered structure, always need one layer of rotations
    dimension += n_qubits

    # Checks if n_qubits is odd or even
    check_odd = n_qubits % 2

    # If n_qubits is even, each layer adds on (n_qubits + (n_qubits - 2)) weights
    # If n_qubits is odd, each layer adds on 2*(n_qubits - 1)
    if check_odd == False:
        dimension += layers*(n_qubits + (n_qubits - 2))
    else:
        dimension += layers*(2*(n_qubits - 1))

    return q_delta * np.random.rand(dimension, requires_grad = True)


def generate_Two_Design(n_qubits:int, layers:int, q_delta:float):
    if n_qubits <= 0:
        raise ValueError("Number of qubits is not a number greater than 0.")
    
    if layers < 0:
        raise ValueError("Number of layers is not a number greater than or equal to 0.")
    
    init_dimension = 0
    dimension = 0

    # For fixed layered structure, always need one layer of rotations
    init_dimension += n_qubits

    # Checks if n_qubits is odd or even
    check_odd = n_qubits % 2

    # If n_qubits is even, each layer adds on (n_qubits + (n_qubits - 2)) weights
    # If n_qubits is odd, each layer adds on 2*(n_qubits - 1)
    if check_odd == False:
        dimension += layers*(n_qubits + (n_qubits - 2))
    else:
        dimension += layers*(2*(n_qubits - 1))

    init_weight_array = q_delta * np.random.randn(init_dimension, requires_grad = True)
    weight_array = q_delta * np.random.randn(dimension, requires_grad = True)

    return qml.SimplifiedTwoDesign(initial_layer_weights = init_weight_array, weights = weight_array, wires = range(n_qubits))






#def b_to_num (n_qubits: int, terms: List(str)):



# This function technically works BUT needs to be tested + optimized further

def A_to_code (idx, ancilla_idx, terms: List[str]):

    if idx < 0:
        raise ValueError("Index of linear combination must be >= 0.")
    
    target_pauli = list(terms[idx])
    
    order_idx = 0

    for i in range(len(target_pauli)):
        if target_pauli[i] == 'I':
            order_idx += 1
            None
        if target_pauli[i] == 'X':
            qml.CNOT(wires = (ancilla_idx, order_idx))
            order_idx += 1
        if target_pauli[i] == 'Y':
            qml.CY(wires = (ancilla_idx, order_idx))
            order_idx += 1
        if target_pauli[i] == 'Z':
            qml.CZ(wires = (ancilla_idx, order_idx))
            order_idx += 1

# This code works, but is adapted from Rigetti code

def A_Ising_num (n: int, zeta: float, eta: float, J: float):
    
    if n <= 0:
        raise ValueError("Number of qubits is not a number greater than 0.")
    
    Acoeffs = [eta/zeta]
    Aterms = ["I" * n]

    Acoeffs += [1/zeta] * n
    xbase = "X" + "I" * (n - 1)
    for ii in range(n, 0, -1):
        Aterms.append(xbase[ii:] + xbase[:ii])

    Acoeffs += [J/zeta] * (n - 1)
    zzbase = "ZZ" + "I" * (n - 2)
    for ii in range(n, 1, -1):
        Aterms.append(zzbase[ii:] + zzbase[:ii])

    return Acoeffs, Aterms
      