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
                # print(f"norm accum ({l*len(c) + lp})")

        return abs(norm)
    
    
    
    # start = time.time()
    mu_sum = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n_qubits):
                # start2 = time.time()
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)
                # print(f"mu sum accum ({l*len(c)*len(c) + lp*len(c) + j})")

    mu_sum = abs(mu_sum)
    # print(f"mu sum: \t{time.time() - start:.2f}s")

    # Cost function C_L
    # start = time.time()
    res = 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))
    # print(f"normalize:\t{time.time() - start:.2f}")
    return res

def cost_global(problem, weights, device, device2):
    """Global version of the cost function. Tends to zero when A|x> is proportional to |b>."""

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
    
    def hadamard_overlap_test(weights, l=None, lp=None, part=None):
        # H on ancilla index
        qml.Hadamard(ancilla_idx * 2)

        # Variational circuit generating a guess for the solution vector |x> applied to the top half
        problem.variational_block(weights, offset=n_qubits)

        # unitary U_b associated to the problem vector |b> applied to the bottom half
        # In this specific example Adjoint(U_b) = U_b.
        problem.U_b()

        # Controlled application of the unitary component A_l of the problem matrix A on the top half.
        problem.CA(l, offset=n_qubits)

        # Controlled application of Adjoint(A_lp) applied to the bottom half
        # In this specific example Adjoint(A_lp) = A_lp. #TODO: is it really?
        problem.CA(lp)

        if part == "Im":
            qml.RZ(phi=-np.pi/2, wires=ancilla_idx*2)

        # CX going from top of top half to top of bottom half, and bottom of top half to bottom of bottom half
        qml.CNOT(wires=(n_qubits*2 - 1, n_qubits-1))
        qml.CNOT(wires=(n_qubits, 0))

        # 2 more hadamards
        qml.Hadamard(wires=[n_qubits*2, n_qubits*2 - 1, n_qubits])

        # we need to measure all the qubits I think? TODO confirm this fact
        return qml.expval(qml.PauliZ(wires=list(range(ancilla_idx*2))))

    hadamard_overlap_test = qml.QNode(hadamard_overlap_test, device2, interface="autograd")

    def gamma(weights, l=None, lp=None):
        gamma_real = hadamard_overlap_test(weights, l=l, lp=lp, part="Re")
        gamma_imag = hadamard_overlap_test(weights, l=l, lp=lp, part="Im")
        # print(f"gamma: {time.time() - start:.2f}")

        return gamma_real + 1.0j * gamma_imag

    norm = 0.0
    overlap = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            # start = time.time()
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)
            # print(f"norm accum ({l*len(c) + lp})")

            overlap = overlap + c[l] * np.conj(c[lp]) * gamma(weights, l, lp)

    norm = abs(norm)


        



def calc_err(n_qubits: int, cost: float, cond_number: float) -> float:
    return np.sqrt(abs(n_qubits * cost * (cond_number ** 2)))