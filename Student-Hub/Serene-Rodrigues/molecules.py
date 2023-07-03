import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import GPyOpt


# Molecules needed are H2, LiH, H3
# still no idea how to read bondlength off of chart
H2_data = qml.data.load("qchem", molname = "H2", basis = "STO-3G", bondlength = 1.1) # 4 qubits
LiH_data = qml.data.load("qchem", molname = "LiH", basis = "STO-3G", bondlength = 1.1) # 4 qubits
H3_data = qml.data.load("qchem", molname = "H3", basis = "STO-3G", bondlength = 1.1) # 4 qubits

# quantum circuit with 4 qubits, all return the expectation of the hamiltonian
dev = qml.device("default.qubit", wires = 4)
@qml.qnode(dev)
def H2_circuit():
    qml.BasisState(H2_data.hf_state, wires = [0,1,2,3])
    for op in H2_data.vqe_gates:
        qml.apply(op)
    return qml.expval(H2_data.hamiltonian)

@qml.qnode(dev)
def LiH_circuit():
    qml.BasisState(LiH_data.hf_state, wires = [0,1,2,3])
    for op in LiH_data.vqe_gates:
        qml.apply(op)
    return qml.expval(LiH_data.hamiltonian)

@qml.qnode(dev)
def H3_circuit():
    qml.BasisState(H3_data.hf_state, wires = [0,1,2,3])
    for op in H3_data.vqe_gates:
        qml.apply(op)
    return qml.expval(H3_data.hamiltonian)

# harttree fox state
hf = qchem.hf_state(electrons = 2, orbitals = 6)

# starting with VQE, ansatz
def ansatz(params):
    qml.BasisState(hf, wires = range(4))
    qml.DoubleExcitation(params[0], wires=[0,1,2,3])
    qml.DoubleExcitation(params[1], wires=[0,1,2,3])

# cost function
@qml.qnode(dev)
def cost_function(params, data):
    ansatz(params)
    return qml.expval(data.hamiltonian)

# now how to mix VQE finding energy and bayesian optimization 
def vqe_energy(circuit, params):
    return circuit(params)
# so for example vqe_energy(H2_circuit, params)

# then the VQE way would be to find the energy this way...
max_iterations = 20

def finding_energy(max_iterations, theta, angle):
    opt = qml.GradientDescentOptimizer(stepsize = 0.4)
    theta = np.array([0.0,0.0], requires_grad = True)

    energy = [cost_function(theta)]
    angle = [theta]
    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_function, theta)

        energy.append(cost_function(theta))
        angle.append(theta)

        # every two steps printing what the energy is
        if n%2 == 0:
            print(f"Step = {n}, Energy = {energy[-1]:.8f} Ha")
    return angle
# do this for all three molecules, but how to integrate them so they are sharing answers and find cost functions
# that use each others measurements

# Bayesian optimization
# how to define target hamiltonians?
# then using GPyOpt like the paper mentions
# objective is the target Hamiltonian I believe?
# bounds = ?
# opt = GPyOpt.methods.BayesianOptimization(f=objective, domain=bounds)
# opt.run_optimization(max_iter=10)
# best_params = opt.x_opt
# best_value = opt.fx_opt