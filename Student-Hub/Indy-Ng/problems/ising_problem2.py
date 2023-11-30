import pennylane as qml
from pennylane import numpy as np

from .problem_base import Problem
from .vqls import generate_H_Ising, A_to_code, b_to_num, reshape_weights

class IsingProblem2(Problem):
    def __init__(self, n_qubits, J, cond_num):
        c, A_terms, self.zeta, self.eta = generate_H_Ising(n_qubits, J, cond_num)
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

    def CA(self, idx):
        A_to_code(idx, ancilla_idx=self.ancilla_idx, terms=self.A_terms)

    def variational_block(self, param):
        #####hardware efficient ansatz
        def U_ENT(wires):
            """
            we will follow the description in Larocca et al. extrapoloated from Fg. 4 and page 23. 
            ZZ entangling gates are applied to neighbouring qubits, with a lattice rather than periodic set-up
            FOR NOW, USE REGULAR CNOT INSTEAD OF exp(i\pi/2Z\otimesZ)
            """
            sigmaX=np.array([[0,1],[1, 0]])
            sigmaZ=np.array([[1,0],[0, -1]])
            # U=np.array(sigmaX)
            num_qubits=len(wires)
            for j in range(num_qubits-1):
                qml.IsingZZ(np.pi, [j, (j+1)])
                # qml.ControlledQubitUnitary(U, 1)
        
        def kandala_circuit(param, wires, d):
            ###simplified circuit from http://arxiv.org/abs/1704.05018

            ###indexing looks a bit fucked up bc its a 1d list (should be easier for pennylane basics)
            ###as a 3d array the indexing would be [d iteration, qubit number, R number]
            ###given $\theta_{j i}^q$ $j\in{1, 2, 3}$
            ###as a 1d list the sequence is [\theta_{00}^0, \theta_{10}^0, \theta_{20}^0, \theta_{01}^0...]
            ###all zeros state
            # qml.BasisState(np.zeros(len(wires)), wires=wires) # shouldn't need this because by default we have all 0s
            ###apply the first set of Euler rotations, without RZ terms
            indtrack=0
            for q in range(len(wires)):
                qml.RX(param[indtrack], wires=[q])
                qml.RZ(param[indtrack+1], wires=[q])
                indtrack=indtrack+2
            
            for i in range(d):
                U_ENT(wires)
                for q in range(len(wires)):
                    #print('qubit',q,  indtrack)
                    qml.RZ(param[indtrack], wires=[q])
                    qml.RX(param[indtrack+1], wires=[q])
                    qml.RZ(param[indtrack+2], wires=[q])
                    indtrack=indtrack+3

        kandala_circuit(param, range(self.n_qubits), self.n_layers)

    # what do I actually want to achieve with this func
    def get_A_and_b(self):
        b = b_to_num(self)

        return self.A_num, b