import pennylane as qml
from pennylane import numpy as np

from .problem_base import Problem
from .vqls import A_Ising_num, A_to_num, A_to_code, b_to_num, reshape_weights

class IsingProblem(Problem):
    def __init__(self, n_qubits, J, zeta, eta_ising):
        c, A_terms = A_Ising_num(n_qubits, zeta, eta_ising, J)
        super().__init__(n_qubits, c, A_terms)

    # getters
    # TEMP
    def get_condition_number(self):
        return np.linalg.norm(self.A_num) * np.linalg.norm(np.linalg.inv(self.A_num))
    
    def get_coeffs(self):
        return self.c, self.A_terms
    
    def get_n_qubits(self):
        return self.n_qubits

    # circuit components
    def U_b(self):
        [qml.Hadamard(wires=i) for i in range(self.n_qubits)]

    def CA(self, idx):
        A_to_code(idx, ancilla_idx=self.ancilla_idx, terms=self.A_terms)

    def variational_block(self, weights):
        # could be put in the state but idk
        layers = 2
        n_parameters = self.n_qubits + layers*(2*self.n_qubits - 2)

        init_weights, weights = reshape_weights(self.n_qubits, n_parameters, layers, weights)

        qml.templates.SimplifiedTwoDesign(
            initial_layer_weights=init_weights,
            weights=weights,
            wires=range(self.n_qubits),
        )

    # what do I actually want to achieve with this func
    def get_A_and_b(self):
        b = b_to_num(self)

        return self.A_num, b