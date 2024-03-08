# %%
import pennylane as qml
from pennylane import numpy as np

import matplotlib.pyplot as plt
# from problems.ising_problem2 import IsingProblem2
from problems.ising_problem import IsingProblem
from problems.toy_benchmark_problem import ToyProblem
from qiskit.providers.fake_provider import *
from bin.cost_function import cost_loc, calc_err, cost_global, local_hadamard_test, hadamard_overlap_test
from bin.hyperparameters import *
from bin.error_mitigation import mitigate_node

import json
import time

# %%
# # Import the library that you need in order to use your IBM account
# from qiskit_ibm_provider import IBMProvider
# from secrets import *


# try:
#     IBMProvider()
# except:
#     IBMProvider.save_account(token=IBM_token, overwrite=True)

# %%
# provider = IBMProvider()
# provider.backends()

# %%
def configured_backend():
    # backend = provider.get_backend("ibm_osaka") # uncomment this line to use a real IBM device
    backend = FakeManila()
    # backend.options.update_options(...)
    return backend

for i in range(1):
    print(f"running trial {i}...")
    # %%
    # define the problem we want to perform our experiment on
    n_qubits = 4
    error = 1.6 * 10**-3
    cond_num = 2

    problem = ToyProblem(n_qubits=n_qubits)
    # problem = IsingProblem(n_qubits=n_qubits, J=0.1, cond_num=cond_num)

    # %%
    # initialize weights
    w = q_delta * np.random.randn(problem.param_shape, requires_grad=True)
    w1 = w.copy()
    init_weights = list(w.copy().numpy())
    # w = q_delta * np.random.randn(batch_size, problem.param_shape, requires_grad=True)

    # %%
    # create our devices to run our circuits on
    # dev_mu = qml.device("default.qubit", wires=n_qubits+1)
    dev_gamma = qml.device("default.qubit", wires=n_qubits*2 + 1)
    dev_mu = qml.device("qiskit.remote", wires=n_qubits+1, backend=configured_backend()) # device for real IBM devices noisy simulators

    # %%
    # in order to make the error mitigation work, we have to pull out the QNode definition outside of the cost function so I'm doing it here
    local_hadamard_test = qml.QNode(local_hadamard_test, dev_mu, interface="autograd")
    # local_hadamard_test = mitigate_node(local_hadamard_test)

    # %%
    # # use non-ML optimization methods
    from scipy.optimize import minimize

    cost_history2 = []

    def cost_fun(w):
        cost = cost_loc(problem, w, local_hadamard_test)
        cost_history2.append(cost)
        # w1s_nonML.append(w[0])
        # w2s_nonML.append(w[1])

        return cost

    OPTIMIZER = "COBYLA"
    if OPTIMIZER == "COBYLA":
        start = time.time()
        res = minimize(cost_fun,
                        w1,
                        method='COBYLA',
                        tol=(error**2)/(n_qubits * cond_num**2)
                        )
        TTS = time.time() - start

        w1 = res.x
        calc_err(n_qubits, cost_history2[-1], cond_num)

    # %%
    from IPython.display import clear_output

    opt = qml.GradientDescentOptimizer(eta)
    # opt = qml.AdagradOptimizer(eta)
    # opt = qml.AdamOptimizer(eta) # TODO: tune decay terms

    cost_history = []

    err = float("inf")
    it = 1

    # %%
    # # training loop

    # best_err = 1000
    # best_w = w
    # # prev_w = w

    # start = time.time()
    # while err > error:
    # # for it in range(steps):
    #     # w, cost = opt.step_and_cost(cost_agg, w)
    #     w, cost = opt.step_and_cost(lambda w: cost_loc(problem, w, local_hadamard_test), w)
    #     # w, cost = opt.step_and_cost(lambda w: cost_global(problem, w, local_hadamard_test, hadamard_overlap_test), w)

    #     err = calc_err(n_qubits, cost, cond_num)
    #     if err < best_err:
    #         best_err = err
    #         best_w = w
        
    #     clear_output(wait=True)

    #     # print(np.array_equal(best_w, w))

    #     print("Step {:3d}       Cost_L = {:9.7f} \t error = {:9.7f}".format(it, cost, err), flush=True)
    #     cost_history.append(cost)

    #     prev_w = w

    #     it += 1

    # print(f"Training time: {time.time() - start}s")

    # %%
    # best_err

    # %% [markdown]
    # ### Optimization studies:
    # - each epoch is taking ~ 20s
    # - Each $\mu$ calculation takes ~0.05s but occasionally spikes to ~0.3s
    # - Each $\mu_{sum}$ has to loop over each combination of A_l for each qubit -> $c^2n$ operations -> ~15s
    # - Each $|\psi|$ also has to loop over each combination, but we're unable to recycle computation because these don't apply $CZ$ like the above -> ~4s
    # - The ising problem has 8 C's
    #     - Note that `len(c) = n_qubits * 2` by definition of H_ising
    # - 8 * 8 * 4 * 2 = ~512 mu accumulations
    # 
    # --> $8n^3 + 4n^2$ iterations
    # 
    # It should be possible to multithread the calculation, but not sure if that's feasible when testing on an actual QC
    # 
    # 
    # * n_qubits = 5: 1100 mu accumulations
    # * n_qubits = 6: 1872 mu accumulations
    # * n_qubits = 7: 2940 mu accumulations

    # %%
    # plt.style.use("seaborn")
    # # plt.plot(np.log(cost_history), "g")
    # # plt.plot(cost_history, "g")
    # plt.plot(cost_history2, "g")
    # plt.ylabel("Cost function")
    # plt.xlabel("Optimization steps")
    # plt.show()

    # %% [markdown]
    # Qualitatively, it's converging slower than the toy problem, suggesting more iterations are needed

    # %%
    from bin.inference import get_cprobs, get_qprobs
    c_probs = get_cprobs(problem)

    # dev_x = qml.device("qiskit.remote", wires=n_qubits, backend=configured_backend())
    dev_x = qml.device("default.qubit", wires=n_qubits, shots=n_shots)

    def prepare_and_sample(problem, weights):

        # Variational circuit generating a guess for the solution vector |x>
        problem.variational_block(weights)

        # We assume that the system is measured in the computational basis.
        # then sampling the device will give us a value of 0 or 1 for each qubit (n_qubits)
        # this will be repeated for the total number of shots provided (n_shots)
        return qml.sample()

    def get_qprobs(problem, w, device):
        sampler = qml.QNode(prepare_and_sample, device)

        # sampler = mitigate_node(sampler)

        raw_samples = sampler(problem, w)
        # raw_samples = np.concatenate(raw_samples, axis=0)# FOR BATCHING

        # convert the raw samples (bit strings) into integers and count them
        samples = []
        for sam in raw_samples:
            samples.append(int("".join(str(bs) for bs in sam), base=2))

        q_probs = np.bincount(samples, minlength=2**problem.n_qubits) / len(raw_samples)
        # q_probs = np.bincount(samples, minlength=2**problem.n_qubits) / n_shots

        return q_probs


    # q_probs = get_qprobs(problem, best_w, dev_x)
    # q_probs2 = get_qprobs(problem, w1, dev_x)

    # # %%
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 4))

    # ax1.bar(np.arange(0, 2 ** n_qubits), c_probs, color="blue")
    # ax1.set_xlim(-0.5, 2 ** n_qubits - 0.5)
    # ax1.set_xlabel("Vector space basis")
    # ax1.set_title("Classical probabilities")

    # ax2.bar(np.arange(0, 2 ** n_qubits), q_probs, color="green")
    # ax2.set_xlim(-0.5, 2 ** n_qubits - 0.5)
    # ax2.set_xlabel("Hilbert space basis")
    # ax2.set_title("Quantum probabilities")

    # ax3.bar(np.arange(0, 2 ** n_qubits), q_probs2, color="green")
    # ax3.set_xlim(-0.5, 2 ** n_qubits - 0.5)
    # ax3.set_xlabel("Hilbert space basis")
    # ax3.set_title("Quantum probabilities")

    # plt.show()

    # %%
    result = {
        "problem": str(problem),
        "n_qubits": n_qubits,
        "ansatz": f"{problem.n_layers}-layer HEA",
        "cost": "local",
        "optimizer": OPTIMIZER,
        "cond_num": cond_num,
        "error_threshold": error,
        "noise_model": NOISE_MODEL,
        "TTS": TTS,
        "STS": len(cost_history2) if OPTIMIZER == "COBYLA" else len(cost_history),
        "final_error": calc_err(n_qubits, cost_history2[-1], cond_num).item(),
        "cost_history": list(cost_history) if OPTIMIZER != "COBYLA" else [tensor.item() for tensor in cost_history2],
        "model_weights": list(w1) if OPTIMIZER == "COBYLA" else list(w.numpy()),
        "init_weights": init_weights
    }

    # %%
    # # serialize to JSON

    with open(f'data/Toy_COBYLA_{NOISE_MODEL}.json', 'a') as fp:
        fp.write(",")
        json.dump(result, fp)

    # %%



