import pennylane as qml
from pennylane import numpy as np
from .problem_base import Problem
from .vqls import generate_H_Ising, A_to_code, b_to_num, reshape_weights

class IsingProblem(Problem):
    def __init__(self, n_qubits, J, cond_num):
        c, A_terms, self.zeta, self.eta = generate_H_Ising(n_qubits, J, cond_num)
        print(A_terms)
        self.n_layers = 4
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

    def CA(self, ancilla_idx, idx):
        A_to_code(idx, ancilla_idx=self.ancilla_idx, terms=self.A_terms)

    def variational_block(self, weights, offset=0):
        # could be put in the state but idk
        layers = 2
        n_parameters = self.n_qubits + layers*(2*self.n_qubits - 2)

        init_weights, weights = reshape_weights(self.n_qubits, n_parameters, layers, weights)

        qml.templates.SimplifiedTwoDesign(
            initial_layer_weights=init_weights,
            weights=weights,
            wires=range(offset, self.n_qubits + offset),
        )

    # what do I actually want to achieve with this func
    def get_A_and_b(self):
        b = b_to_num(self)

        return self.A_num, b