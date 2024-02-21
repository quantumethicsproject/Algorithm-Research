import pennylane as qml
from pennylane import numpy as np
from .problem_base import Problem
from .vqls import generate_H_Ising, A_to_code, b_to_num, reshape_weights

class IsingProblem3(Problem):
    def __init__(self, n_qubits, J, cond_num):
        c, A_terms, self.zeta, self.eta = generate_H_Ising(n_qubits, J, cond_num)
        print(A_terms)
        self.n_layers = 1
        self._param_shape = 5 * self.n_layers * n_qubits
        super().__init__(n_qubits, c, A_terms)


    # getters
    # TEMP
    def get_eta_zeta(self):
        return self.zeta, self.eta
    
    def get_coeffs(self):
        return self.c, self.A_terms
    
    def get_n_qubits(self):
        return self.n_qubits
    
    @property
    def param_shape(self):
        return self._param_shape

    # circuit components
    def U_b(self):
        [qml.Hadamard(wires=i) for i in range(self.n_qubits)]

    def CA(self, idx, offset=0):
        A_to_code(idx, ancilla_idx=self.ancilla_idx, terms=self.A_terms)

    def variational_block(self, weights, offset=0):
        weights_used = 0
        [qml.RY(phi=weights[i + weights_used], wires=i+offset) for i in range(self.n_qubits)]
        weights_used += self.n_qubits

        for _ in range(self.n_layers):
            # # 1 layer
            [qml.CZ(wires=(j, j+1)) for j in range(0, self.n_qubits-1, 2)]
            [qml.RY(phi=weights[i + weights_used], wires=i+offset) for i in range(self.n_qubits//2 * 2)]
            weights_used += self.n_qubits // 2 * 2 - 1
            [qml.CZ(wires=(j, j+1)) for j in range(1, self.n_qubits-1, 2)]
            [qml.RY(phi=weights[i + weights_used], wires=i+offset) for i in range(1,(self.n_qubits + 1) // 2 * 2 -1)]
            weights_used += (self.n_qubits + 1) // 2 * 2 -1

    # what do I actually want to achieve with this func
    def get_A_and_b(self):
        b = b_to_num(self)

        return self.A_num, b