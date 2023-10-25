import pennylane as qml
from hyperparameters import *
from pennylane import numpy as np
from problem_base import Problem

class IndysProblem(Problem):
    # TODO: after creating the A breakdown thing, we can make this more generic by refactoring these static methods into self
    # Coefficients of the linear combination A = c_0 A_0 + c_1 A_1 ...
    c = np.array([1.0, 0.2, 0.2])

    @staticmethod
    def U_b():
        """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
        
        # This loop applies the Hadamard operator on each wire
        for idx in range(n_qubits):
            qml.Hadamard(wires=idx)

    @staticmethod
    def CA(idx):
        """Controlled versions of the unitary components A_l of the problem matrix A."""
        if idx == 0:
            # Identity operation
            None

        elif idx == 1:
            # CNOT gate is the same as controlled Pauli-X gate
            qml.CNOT(wires=[ancilla_idx, 0])
            qml.CZ(wires=[ancilla_idx, 1])

        elif idx == 2:
            qml.CNOT(wires=[ancilla_idx, 0])

    @staticmethod
    def variational_block(weights):
        """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
        # We first prepare an equal superposition of all the states of the computational basis.
        for idx in range(n_qubits):
            qml.Hadamard(wires=idx)

        # A very minimal variational circuit.
        for idx, element in enumerate(weights):
            qml.RY(element, wires=idx)

    @staticmethod
    def get_A_and_b():
        Id = np.identity(2)
        Z = np.array([[1, 0], [0, -1]])
        X = np.array([[0, 1], [1, 0]])

        A_0 = np.identity(8)
        A_1 = np.kron(np.kron(X, Z), Id)
        A_2 = np.kron(np.kron(X, Id), Id)

        A_num = IndysProblem.c[0] * A_0 + IndysProblem.c[1] * A_1 + IndysProblem.c[2] * A_2
        b = np.ones(8) / np.sqrt(8) # Q: why is this equivalent to (H\ket{0})^{\otimes n}??

        return A_num, b
