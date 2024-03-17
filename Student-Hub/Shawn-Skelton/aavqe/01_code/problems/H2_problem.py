import pennylane as qml
from pennylane import numpy as np

from problems.useful_fcns import MOL_H_BUILD, EASY_HAM


class H2_PROBLEM():
    def __init__(self, mol, bdl):
        self.H, self.gsE, self.qubits=MOL_H_BUILD(mol, bdl)
        self.H0=EASY_HAM(self.qubits)

    ####define the functions that get varaibles from self
    def GET_HS(self):
        return self.H, self.H0

    def GET_gsE(self):
        return self.gsE

    def GET_QUBITS(self):
        return self.qubits


    # def get_n_qubits(self):
    #     return self.qubits


