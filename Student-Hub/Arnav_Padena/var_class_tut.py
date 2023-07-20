# import stuff
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import pandas as pd
import math

# TODO: Determine appropriate amount of qubits to use for CANCER dataset
# Read: State Preparation section of paper

def numQubits(data):
    numObvs = data.shape[0] * data.shape[1]
    # find the ceiling of the log of numObvs
    # this is the number of qubits needed
    return math.ceil(math.log(numObvs, 2))

# TODO: Implement non-informative padding procedure described in paper
def padData(data, n_qubits):
    # find the number of 0s to pad with
    numZeros = 2 ** n_qubits - data.shape[0]
    # create array of 0s
    zeros = np.zeros((numZeros, data.shape[1]))
    # concatenate data and zeros
    return np.concatenate((data, zeros))

# define the device
dev = qml.device("default.qubit", wires=4)

# TODO: Implement Amplitude Encoding using Pennylane


# Load the data
path = './wdbc.data'

# Documentation for the col_names can be found in wdbc.names
col_names = ["ID", "diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points", "symmetry", "fractal dimension",
             "SE_radius", "SE_texture", "SE_perimeter", "SE_area", "SE_smoothness", "SE_compactness", "SE_concavity", "SE_concave_points", "SE_symmetry", "SE_fractal dimension",
             "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness", "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry", "worst_fractal dimension"]

data = pd.read_csv(path, names=col_names)
data.head()
data = data.to_numpy()

n_qubits = numQubits(data)
data = padData(data, n_qubits)


#### Everything below this is from tutorial, but isn't necessarily relevant to actual replication of paper
#### This is because the tutorial uses angle encoding while paper uses amplitude encoding
#### Difference is explained here https://pennylane.ai/qml/glossary/quantum_embedding


# define the layer
def layer(W):
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])

# define the state preparation function
def statepreparation(x):
    qml.BasisState(x, wires=[0, 1, 2, 3])

# define the variational circuit
@qml.qnode(dev, interface="autograd")
def circuit(weights, x):
    statepreparation(x)

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))

# define the variational classifier
def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

# define the loss function
def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

# define the accuracy function
def accuracy(labels, predictions):
    loss = 0

    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1

    loss = loss / len(labels)

    return loss

# define the cost function
def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)
