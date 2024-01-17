from pennylane.transforms import mitigate_with_zne
import pennylane as qml
from bin.hyperparameters import *

def mitigate_node(noisy_qnode):
    if not NOISY:
        return noisy_qnode
    
    extrapolate = qml.transforms.richardson_extrapolate
    scale_factors = [1, 2, 3]

    mitigated_qnode = mitigate_with_zne(noisy_qnode, scale_factors, qml.transforms.fold_global, extrapolate)

    return mitigated_qnode