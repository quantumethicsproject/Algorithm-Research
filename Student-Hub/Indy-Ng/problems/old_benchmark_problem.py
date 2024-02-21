import pennylane as qml
from pennylane import numpy as np
from .problem_base import Problem
from .vqls import A_to_num, A_to_code, b_to_num, reshape_weights

class BenchmarkProblemDeprecated(Problem):
    def __init__(self, n_qubits):
        # TODO: make the A_to_code function work with H
        c = [1]
        A_terms = ["X" * n_qubits]
        B_terms = ["H" * n_qubits]

        super().__init__(n_qubits, c, A_terms)

        self.param_shape = n_qubits * 3 # TODO

    def get_coeffs(self):
        return self.c, self.A_terms
    
    def get_n_qubits(self):
        return self.n_qubits
        

    def U_b(self):
        """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
        [qml.Hadamard(wires=i) for i in range(self.n_qubits)]
        
    def CA(self, ancilla_idx, idx, offset=0):
        A_to_code(idx, ancilla_idx=ancilla_idx, terms=self.A_terms, offset=offset)

    def variational_block(self, weights, offset=0):
        weights_used = 0
        [qml.RY(phi=weights[i + weights_used], wires=i+offset) for i in range(self.n_qubits)]
        weights_used += self.n_qubits

        # # 1 layer
        # [qml.CZ(wires=(j, j+1)) for j in range(0, self.n_qubits-1, 2)]
        # [qml.RY(phi=weights[i + weights_used], wires=i+offset) for i in range(self.n_qubits)]
        # weights_used += self.n_qubits
        # [qml.CZ(wires=(j, j+1)) for j in range(1, self.n_qubits-1, 2)]
        # [qml.RY(phi=weights[i + weights_used], wires=i+offset) for i in range(1,self.n_qubits-1)]

    # what do I actually want to achieve with this func
    def get_A_and_b(self):
        b = b_to_num(self)

        return self.A_num, b