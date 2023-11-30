import pennylane as qml
from pennylane import numpy as np

import time

def cost_loc(problem, weights, device):
    """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""

    c, _ = problem.get_coeffs()
    n_qubits = problem.get_n_qubits()
    ancilla_idx = n_qubits

    def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):

        # First Hadamard gate applied to the ancillary qubit.
        qml.Hadamard(wires=ancilla_idx)

        # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
        # phase gate.
        if part == "Im" or part == "im":
            qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)

        # Variational circuit generating a guess for the solution vector |x>
        problem.variational_block(weights)

        # Controlled application of the unitary component A_l of the problem matrix A.
        problem.CA(l)

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
        problem.CA(lp)

        # Second Hadamard gate applied to the ancillary qubit.
        qml.Hadamard(wires=ancilla_idx)

        # Expectation value of Z for the ancillary qubit.
        return qml.expval(qml.PauliZ(wires=ancilla_idx))
    
    local_hadamard_test = qml.QNode(local_hadamard_test, device, interface="autograd")
    
    # Computes the mu coefficients
    def mu(weights, l=None, lp=None, j=None):
        """Generates the coefficients to compute the "local" cost function C_L."""

        # start = time.time()
        mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
        mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")
        # print(f"mu: {time.time() - start:.2f}")

        return mu_real + 1.0j * mu_imag

    def psi_norm(weights):
        """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
        norm = 0.0

        for l in range(0, len(c)):
            for lp in range(0, len(c)):
                # start = time.time()
                norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)
                # print(f"norm accum ({l*len(c) + lp}): {time.time() - start:.3f}")

        return abs(norm)
    
    
    
    # start = time.time()
    mu_sum = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n_qubits):
                # start2 = time.time()
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)
                # print(f"mu sum accum ({l*len(c)*len(c) + lp*len(c) + j}): {time.time() - start2:.3f}")

    mu_sum = abs(mu_sum)
    # print(f"mu sum: \t{time.time() - start:.2f}s")

    # Cost function C_L
    # start = time.time()
    res = 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))
    # print(f"normalize:\t{time.time() - start:.2f}")
    return res

def calc_err(n_qubits: int, cost: float, cond_number: float) -> float:
    return np.sqrt(n_qubits * cost * (cond_number ** 2))