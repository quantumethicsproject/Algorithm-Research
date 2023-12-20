from pennylane import numpy as np

# Number of quantum measurements performed to get a probability distribution of results
n_shots = 10**6

# Number of optimization steps
steps = 100

# Learning rate of optimization algorithm
eta = 3

# Initial spread of random quantum weights
q_delta = 0.001

# Seed for RNG
rng_seed = 0

# exploit param broadcasting
batch_size = 1

NOISY = True