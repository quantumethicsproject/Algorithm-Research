from pennylane import numpy as np

# Number of quantum measurements performed to get a probability distribution of results
n_shots = 10**6

# Number of optimization steps
steps = 30

# Learning rate of optimization algorithm
# TODO: hyperparameter tuning for eta (think coarse to fine)
eta = 4

# Initial spread of random quantum weights
q_delta = 0.001

# Seed for RNG
rng_seed = 0