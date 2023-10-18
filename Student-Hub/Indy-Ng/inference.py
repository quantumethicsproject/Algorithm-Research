import pennylane as qml
# from circuits import variational_block
from circuits import IndysProblem
from pennylane import numpy as np
from hyperparameters import *

def prepare_and_sample(weights):

    # Variational circuit generating a guess for the solution vector |x>
    IndysProblem.variational_block(weights)

    # We assume that the system is measured in the computational basis.
    # then sampling the device will give us a value of 0 or 1 for each qubit (n_qubits)
    # this will be repeated for the total number of shots provided (n_shots)
    return qml.sample()

# TODO: write a function that can automatically convert our A written in terms of linear combinations
#       of A_l into explicit matrix representation using NumPy arrays
def get_cprobs():
    """Returns x as a classical vector"""

    A_num, b = IndysProblem.get_A_and_b()
    A_inv = np.linalg.inv(A_num)
    x = np.dot(A_inv, b)

    c_probs = (x / np.linalg.norm(x)) ** 2

    return c_probs

def get_qprobs(w, device):
    sampler = qml.QNode(prepare_and_sample, device)

    raw_samples = sampler(w)

    # convert the raw samples (bit strings) into integers and count them
    samples = []
    for sam in raw_samples:
        samples.append(int("".join(str(bs) for bs in sam), base=2))

    q_probs = np.bincount(samples) / n_shots

    return q_probs
