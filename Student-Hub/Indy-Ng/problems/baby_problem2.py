import pennylane as qml
from pennylane import numpy as np
from .problem_base import Problem
from .vqls import A_to_num, A_to_code, b_to_num

class UFProblem(Problem):
    def __init__(self):
        n_qubits = 2
        c = [2, -1, -0.5, -0.5]
        A_terms = ["II", "IX", "XY", "YX"]

        super().__init__(n_qubits, c, A_terms)

        self.param_shape = n_qubits

    def get_coeffs(self):
        return self.c, self.A_terms
    
    def get_n_qubits(self):
        return self.n_qubits
        

    def U_b(self):
        """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
        [qml.Hadamard(i) for i in range(self.n_qubits)]
        
    def CA(self, ancilla_idx, idx, offset=0):
        A_to_code(idx, ancilla_idx=ancilla_idx, terms=self.A_terms, offset=offset)

    def variational_block(self, weights, offset=0):
        [qml.RY(phi=weights[i], wires=i+offset) for i in range(self.n_qubits)]

    # what do I actually want to achieve with this func
    def get_A_and_b(self):
        b = b_to_num(self)

        return self.A_num, b