import pennylane as qml
from pennylane import numpy as np
from .problem_base import Problem
from .vqls import A_to_num, A_to_code, b_to_num, reshape_weights

class ToyProblem(Problem):
    def __init__(self, n_qubits):
        # TODO: make the A_to_code function work with H
        if n_qubits == 1:
            # c = [1]
            # A_terms = ["H"]
            # B_terms = ["X"]
            
            c = [1, 0.25]
            A_terms = ["I", "Z"]
            B_terms = ["X"]
            pass

        if n_qubits == 2:
            # c = [1]
            # A_terms = ["XH"]
            # B_terms = ["HH"]

            c = [1, 0.25]
            A_terms = ["II", "IZ"]
            B_terms = ["HI"]

            # # actually stupid problem
            # c = [1]
            # A_terms = ["IZ"]
            # B_terms = ["IH"]

        if n_qubits == 3:
            c = [1, 0.25]
            A_terms = ["III", "IIZ"]
            B_terms = ["HHI"]

        if n_qubits == 5:
            c = [1, 0.2, 0.2]
            A_terms = ["IIIII", "XZIII", "XIIII"]
            B_terms = ["HIHHH"]

            # c = [1, 0.25]
            # A_terms = ["IIIII", "IIIIX"]
            # B_terms = ["HHHHH"]

        super().__init__(n_qubits, c, A_terms)

        self.param_shape = n_qubits * 3 # TODO

    def get_coeffs(self):
        return self.c, self.A_terms
    
    def get_n_qubits(self):
        return self.n_qubits
        

    def U_b(self):
        """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
        if self.n_qubits == 1:
            qml.PauliX(0)
        if self.n_qubits == 2:
            pass
            # qml.PauliX(0)
            qml.Hadamard(0)
            # [qml.Hadamard(wires=i) for i in [0,1]]
        if self.n_qubits == 3:
            [qml.Hadamard(wires=i) for i in [0,1]]
        if self.n_qubits == 5:
            [qml.Hadamard(wires=i) for i in [0,2,3,4]]
        
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