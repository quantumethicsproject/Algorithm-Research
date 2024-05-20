import pennylane as qml
from pennylane import numpy as np
from .problem_base import Problem
from .vqls import A_to_num, A_to_code, b_to_num, reshape_weights
from scipy.optimize import fsolve

def get_phi2(k, phi1):
    func = lambda phi2 : k - (np.cos(2*phi1 + phi2)/np.cos(2*phi1 - phi2)) 

    phi2_initial_guess = 0.5
    solution = fsolve(func, phi2_initial_guess)

    return solution

def c0not(ctrl, target):
    # cnot conditioned on 0
    qml.PauliX(wires=ctrl)
    qml.CNOT(wires=(ctrl, target))
    qml.PauliX(wires=ctrl)

def U_a(wires, ancilla_idx):
    [qml.Hadamard(wires=i) for i in wires]
    qml.Hadamard(ancilla_idx)

def tunable_A(phi1, phi2, n_qubits, anc1, anc2):
    qml.Hadamard(wires=anc2)
    c0not(anc1, anc2)

    # Z rotation by phi1
    qml.RZ(phi1, wires=anc2)

    c0not(anc1, anc2)
    # U_a acts on all wires + first ancilla
    U_a(wires=range(n_qubits), ancilla_idx=anc1)
    c0not(anc1, anc2)

    # Z rotation by phi2
    qml.RZ(phi2, wires=anc2)

    c0not(anc1, anc2)
    U_a(wires=range(n_qubits), ancilla_idx=anc1) # hadamards are hermititian so conjugate = itself
    c0not(anc1, anc2)

    # Z rotation by phi2
    qml.RZ(phi1, wires=anc2)
    
    c0not(anc1, anc2)
    qml.Hadamard(wires=anc2)

class BenchmarkProblem(Problem):
    def __init__(self, n_qubits, cond_num=2):
        # TODO: make the A_to_code function work with H
        c = [1]
        A_terms = ["X" * n_qubits]
        B_terms = ["H" * n_qubits]

        super().__init__(n_qubits, c, A_terms)

        self.param_shape = n_qubits # TODO
        # self.param_shape = n_qubits * 3 # TODO

        self.anc1 = self.n_qubits + 1
        self.anc2 = self.n_qubits + 2

        self.phi1 = 0.5
        self.phi2 = get_phi2(cond_num, self.phi1)

        A_func = qml.matrix(tunable_A)
        self.A_unitary = A_func(self.phi1, self.phi2, self.n_qubits, self.anc1, self.anc2)

    def get_coeffs(self):
        return self.c, self.A_terms
    
    def get_n_qubits(self):
        return self.n_qubits
        

    def U_b(self):
        """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
        [qml.Hadamard(wires=i) for i in range(self.n_qubits)]

    # destroy the ancilla qubits
    # qml.measure(wires=anc1, reset=True)
    # qml.measure(wires=anc2, reset=True)
        
    def CA(self, ancilla_idx, idx, offset=0):
        qml.ControlledQubitUnitary(self.A_unitary, control_wires=ancilla_idx, wires=[i for i in range(self.anc2 + 1) if i != ancilla_idx])

        # reset the ancilla qubits
        qml.measure(self.anc1, reset=True)
        qml.measure(self.anc2, reset=True)

    def variational_block(self, weights, offset=0):
        weights_used = 0
        [qml.RY(phi=weights[i + weights_used], wires=i+offset) for i in range(self.n_qubits)]
        weights_used += self.n_qubits

        # # # 1 layer
        # [qml.CZ(wires=(j, j+1)) for j in range(0, self.n_qubits-1, 2)]
        # [qml.RY(phi=weights[i + weights_used], wires=i+offset) for i in range(self.n_qubits)]
        # weights_used += self.n_qubits
        # [qml.CZ(wires=(j, j+1)) for j in range(1, self.n_qubits-1, 2)]
        # [qml.RY(phi=weights[i + weights_used], wires=i+offset) for i in range(1,self.n_qubits-1)]

    # what do I actually want to achieve with this func
    def get_A_and_b(self):
        b = b_to_num(self)

        return self.A_num, b