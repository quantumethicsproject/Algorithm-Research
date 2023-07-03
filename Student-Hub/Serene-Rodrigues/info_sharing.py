# Imports
import pennylane as qml
import numpy as np
from scipy.optimize import minimize
from pennylane.templates import AngleEmbedding, BasicEntanglerLayers
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt


# circuit for H2
def vqe_circuit(params):
    qml.templates.AngleEmbedding(params[:4], wires=[0,1,2,3])
    qml.templates.BasicEntanglerLayers(params[4:], wires=[0,1,2,3])
    return qml.expval(qml.PauliZ(0)@qml.PauliZ(1)@qml.PauliZ(2)@qml.PauliZ(3))

# gives the vqe circuit
def potential_energy(params):
    return vqe_circuit(params)

# goal of optimization to find the ground state energy
def bayesian_optimization_objective(params):
    return -potential_energy(params)

def bayesian_optimization():
    # this would be regular bayesian_optimization for one variable
    N = 10
    points = 5
    # need bounds
    # bounds = ?
    # to start somewhere
    # points_init = np.random.uniform(low=0, high=?, size = (points, ?))
    new_energy = np.inf
    params_2 = None

    for i in range(N):
        params = []
        # for point in points_init:
        #     result = want to minimize bayesian optimization step
        #     params.append(result.x)
        #     energies = [potential_energy(p) for p in params]
        #     min_energy = np.argmin(energies)
        #     energy = energies[min_energy]
        #     params_2 = params[min_energy]

        # if energy < new_energy:
        #         new_energy = energy

        #print(print each step)

    # print(# best energy with best params)