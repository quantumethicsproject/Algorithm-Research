from abc import ABC, abstractmethod
import numpy as np
from .vqls import A_to_num

class Problem(ABC):
    def __init__(self, n_qubits, c, A_terms) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.A_num = A_to_num(n_qubits, c, A_terms)
        self.A_terms = A_terms

        # normalize c
        self.c = np.array(c) / np.linalg.norm(self.A_num, ord=2)

        # Total number of qubits; here we add an ancillary qubit
        self.tot_qubits = self.n_qubits + 1
        # Index of ancillary qubit (Python lists are 0-indexed)
        self.ancilla_idx = self.n_qubits

    @abstractmethod
    def get_coeffs():
        """gets c, A_l"""
        pass
    
    @abstractmethod
    def get_n_qubits():
        """gets number of qubits of your problem"""
        pass

    @abstractmethod
    def U_b():
        """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
        pass

    @abstractmethod
    def CA(idx):
        """controlled application of A_l"""
        pass

    @abstractmethod
    def variational_block(weights):
        """function that defines the ansatz"""
        pass

    @abstractmethod
    def get_A_and_b():
        """returns classical A and b"""
        pass

