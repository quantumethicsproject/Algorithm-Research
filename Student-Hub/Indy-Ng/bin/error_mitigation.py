from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory
from pennylane.transforms import mitigate_with_zne
from bin.hyperparameters import *

def mitigate_node(noisy_qnode):
    if not NOISY:
        return noisy_qnode
    
    extrapolate = RichardsonFactory.extrapolate
    scale_factors = [1, 2, 3]

    mitigated_qnode = mitigate_with_zne(noisy_qnode, scale_factors, fold_global, extrapolate)

    return mitigated_qnode