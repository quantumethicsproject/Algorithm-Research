from pennylane import numpy as np

# Number of system qubits; this determines the size of the matrix A and vector b
n_qubits = 3

# Number of quantum measurements performed to get a probability distribution of results
n_shots = 10**6

# Total number of qubits; here we add an ancillary qubit
tot_qubits = n_qubits + 1

# Index of ancillary qubit (Python lists are 0-indexed)
ancilla_idx = n_qubits

# Number of optimization steps
steps = 30

# Learning rate of optimization algorithm
# TODO: hyperparameter tuning for eta (think coarse to fine)
eta = 0.8

# Initial spread of random quantum weights
q_delta = 0.001

# Seed for RNG
rng_seed = 0