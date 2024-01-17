import pennylane as qml
from pennylane import numpy as np

import time

def local_hadamard_test(weights, problem, l=None, lp=None, j=None, part=None):
    """this function implements the local hadamard test for calculating mu and the norm"""

    ancilla_idx = problem.get_n_qubits()

    # First Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=ancilla_idx)

    # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
    # phase gate.
    if part == "Im" or part == "im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)

    # Variational circuit generating a guess for the solution vector |x>
    problem.variational_block(weights)

    # Controlled application of the unitary component A_l of the problem matrix A.
    problem.CA(ancilla_idx, l)

    # Adjoint of the unitary U_b associated to the problem vector |b>.
    # In this specific example Adjoint(U_b) = U_b.
    problem.U_b()

    # Controlled Z operator at position j. If j = -1, apply the identity.
    if j != -1:
        qml.CZ(wires=[ancilla_idx, j])

    # Unitary U_b associated to the problem vector |b>.
    problem.U_b()

    # Controlled application of Adjoint(A_lp).
    # In this specific example Adjoint(A_lp) = A_lp.
    problem.CA(ancilla_idx, lp)

    # Second Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=ancilla_idx)

    # Expectation value of Z for the ancillary qubit.
    return qml.expval(qml.PauliZ(wires=ancilla_idx))

# Computes the mu coefficients
def mu(weights, local_hadamard_test, problem, l=None, lp=None, j=None):
    """Generates the coefficients to compute the "local" cost function C_L."""

    # start = time.time()
    mu_real = local_hadamard_test(weights, problem, l=l, lp=lp, j=j, part="Re")
    mu_imag = local_hadamard_test(weights, problem, l=l, lp=lp, j=j, part="Im")
    # print(f"mu: {time.time() - start:.2f}")

    return mu_real + 1.0j * mu_imag

def get_bin(state: int, n_qubits):
    """
    Helper function that identifies the correct bin for the overlap test. Details can be found in Cincio et. al

    @param
    state: a measurement outcome as an int 
    return: (-1 or 1, corresponding to whether the prob on the bitstring should be added or subtracted)
    """
    acc = 1

    # if aux qubit is 1
    if state & 2**(n_qubits*2):
        acc *= -1

    for i in range(n_qubits):
        if state & 2**i and state & 2**(i + n_qubits):
            acc *= -1

    return acc

def gamma(weights, hadamard_overlap_test, problem, l=None, lp=None):
    """calculates the gamma coefficients for C_G"""
    n_qubits = problem.get_n_qubits()

    probs_real = hadamard_overlap_test(weights, problem, l=l, lp=lp, part="Re")
    probs_imag = hadamard_overlap_test(weights, problem, l=l, lp=lp, part="Im")

    gamma_real = 0
    gamma_imag = 0

    # I have a feeling a lot of these are cancelling each other out resulting in a very low output value
    for state, prob in enumerate(probs_real):
        gamma_real += get_bin(state, n_qubits) * prob
    
    for state, prob in enumerate(probs_imag):
        gamma_imag += get_bin(state, n_qubits) * prob

    # print(f"gamma: {time.time() - start:.2f}")

    return 2 * (gamma_real + 1.0j * gamma_imag) # see appendix C for the 2x coeff

def psi_norm(weights, c, local_hadamard_test, problem):
    """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
    norm = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            # start = time.time()
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, local_hadamard_test, problem, l, lp, -1)
            # print(f"norm accum ({l*len(c) + lp})")

    return abs(norm)

def hadamard_overlap_test(weights, problem, l=None, lp=None, part=None):
    """implements the overlap test for C_G"""

    n_qubits = problem.get_n_qubits()
    ancilla_idx = n_qubits * 2

    # H on ancilla index
    qml.Hadamard(ancilla_idx)

    # Variational circuit generating a guess for the solution vector |x> applied to the top half
    problem.variational_block(weights, offset=n_qubits)

    # unitary U_b associated to the problem vector |b> applied to the bottom half
    # In this specific example Adjoint(U_b) = U_b.
    problem.U_b()

    # Controlled application of the unitary component A_l of the problem matrix A on the top half.
    problem.CA(ancilla_idx, l, offset=n_qubits)

    # Controlled application of Adjoint(A_lp) applied to the bottom half
    # In this specific example Adjoint(A_lp) = A_lp. #TODO: is it really?
    problem.CA(ancilla_idx, lp)

    if part == "Im":
        qml.RZ(phi=-np.pi/2, wires=ancilla_idx)

    # bell basis observable
    [qml.CNOT(wires=(i+n_qubits, i)) for i in range(n_qubits)]
    [qml.Hadamard(wires=i) for i in range(n_qubits, n_qubits*2 + 1)]

    # to get P(0) - P(1) we need to perform linear classical post-processing which involves using the probabilities
    return qml.probs(wires=range(n_qubits*2 + 1))

def cost_loc(problem, weights, local_hadamard_test):
    """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""

    c, _ = problem.get_coeffs()
    n_qubits = problem.get_n_qubits()
    
    # start = time.time()
    mu_sum = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n_qubits):
                # start2 = time.time()
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, local_hadamard_test, problem, l, lp, j)
                # print(f"mu sum accum ({l*len(c)*len(c) + lp*len(c) + j})")

    mu_sum = abs(mu_sum)
    # print(f"mu sum: \t{time.time() - start:.2f}s")

    # Cost function C_L
    # start = time.time()
    res = 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights, c, local_hadamard_test, problem))
    # print(f"normalize:\t{time.time() - start:.2f}")
    return res

def cost_global(problem, weights, local_hadamard_test, hadamard_overlap_test):
    """Global version of the cost function. Tends to zero when A|x> is proportional to |b>."""

    c, _ = problem.get_coeffs()

    norm = 0.0
    overlap = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            # start = time.time()
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, local_hadamard_test, problem, l, lp, -1)
            # print(f"norm accum ({l*len(c) + lp})")

            overlap = overlap + c[l] * np.conj(c[lp]) * gamma(weights, hadamard_overlap_test, problem, l, lp)

    norm = abs(norm)
    overlap = abs(overlap)

    return 1 - overlap / norm # TODO: double check this expression


def calc_err(n_qubits: int, cost: float, cond_number: float) -> float:
    """helper function that turns a cost value into an error bound"""
    return np.sqrt(abs(n_qubits * cost * (cond_number ** 2)))