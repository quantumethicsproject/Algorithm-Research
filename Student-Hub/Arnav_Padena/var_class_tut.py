# import stuff
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import pandas as pd

# define the device
dev = qml.device("default.qubit", wires=4)

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

# load the data
column_names = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal dimension"]

data_list =[]
file = "wdbc.data"
df = pd.read_csv(file, names = column_names)
print(df)

