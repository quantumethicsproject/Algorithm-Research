
"""
qRBM
"""
import copy

# Modules
import pennylane as qml
from pennylane import numpy as np
from pennylane import qaoa

# Quantum Boltzmann Machine
class qRBM:
    # Constructor
    def __init__(self, num_visible, num_hidden, qaoa_steps=1):
        self.visible_nodes = num_visible
        self.hidden_nodes = num_hidden
        self.total_nodes = self.visible_nodes + self.hidden_nodes
        self.qaoa_steps = qaoa_steps

        # Still need to figure out where this comes from
        # Setting up the angle; used in initial state for QAOA
        self.beta_temp = 2.0                                                      # can make this a passable parameter?
        self.state_prep_angle_phi = np.arctan(np.exp(-self.beta_temp / 2.0)) * 2.0

        # Initialize the weights to something random
        # add var for endpoints?
        self.weights = np.asarray(
            np.random.uniform(
                low=-0.1 * np.sqrt(6 / self.total_nodes),
                high=0.1 * np.sqrt(6 / self.total_nodes),
                size=(num_visible, num_hidden)
            )
        )

        # Initialize the biases to something random
        # TO BE ADDED


        # NOTE:
        # W[i][j] = weight between i-th visible node and j-th hidden node


        # Setting up cost and mixer Hamiltonians
        # Indices
        # Note that indices of visible nodes < indices of hidden nodes
        visible_indices = [i for i in range(self.visible_nodes)]
        hidden_indices = [i + self.visible_nodes for i in range(self.hidden_nodes)]
        total_indices = [i for i in range(self.total_nodes)]

        # Setting up the full cost Hamiltonian
        coeffs_HC = np.array([])
        observables_HC = []
        for i in visible_indices:
            for j in hidden_indices:
                coeffs_HC = np.append(coeffs_HC, -1 * self.weights[i][j - self.visible_nodes])
                observables_HC.append(qml.PauliZ(i) @ qml.PauliZ(j))

        self.full_HC = qml.Hamiltonian(coeffs_HC, observables_HC)

        # Setting up the full mixer Hamiltonian
        coeffs_HM = np.array([])
        observables_HM = []
        for i in total_indices:
            coeffs_HM = np.append(coeffs_HM, 1.0)
            observables_HM.append(qml.PauliX(i))

        self.full_HM = qml.Hamiltonian(coeffs_HM, observables_HM)


    # QAOA cost and mixer layers
    def unclamped_qaoa_layer(self, gamma, nu):
        qaoa.cost_layer(gamma, self.full_HC)
        qaoa.mixer_layer(nu, self.full_HM)


    # Unclamped QAOA circuit
    def unclamped_circuit(self, params, **kwargs):
        for w in range(self.total_nodes):
            # this part doesn't work i forgor
            qml.RX(phi=self.state_prep_angle_phi, wires=w+self.total_nodes)         # bro WHAT...... HOW
            qml.CNOT([w+self.total_nodes, w])

        qml.layer(self.unclamped_qaoa_layer, self.qaoa_steps, params[0], params[1])
        gamma_prime = qml.expval(qml.PauliZ(0))
        nu_prime = qml.expval(qml.PauliZ(1))

        return gamma_prime, nu_prime


    # Sigmoid Helper Function
    # Converts integers to probabilities (in range(0,1))
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


    # Training arc
    def train(self, training_data, learning_rate=0.1, n_epochs=100, quantum_percentage=1.0, classical_percentage=0.0):    #quantum, classical percentage omitted

        data = np.asarray(training_data)

        visible_indices = [i for i in range(self.visible_nodes)]
        hidden_indices = [i + self.visible_nodes for i in range(self.hidden_nodes)]

        # Initialize nu, gamma as random
        unc_gammas = 0.5
        unc_nus = 0.5

        for epoch in range(n_epochs):
            print("Beginning epoch no.", epoch)

            new_weights = copy.deepcopy(self.weights)

            # QAOA to optimize gamma and nu
            unc_gammas, unc_nus = self.unclamped_circuit(unc_gammas, unc_nus)

            neg_phase_quantum = np.zeros_like(self.weights)

            # Classically optimizer to figure out updates to next epoch of gamma, nu
            # VQE in example

            hidden_probs = self.sigmoid(np.dot(data, self.weights))
            pos_phase = np.dot(data.T, hidden_probs) * (1./float(len(data)))

            pos_hidden_states = hidden_probs > np.random.rand(len(data), self.hidden_nodes)

            # Which nodes get activated
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self.sigmoid(neg_visible_activations)               # convert number to probability
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self.sigmoid(neg_hidden_activations)

            neg_phase_classical = np.dot(neg_visible_probs.T, neg_hidden_probs) * 1./len(data)

            # Update node weights w/ weighted avg of quantum and classical
            new_weights += learning_rate * (pos_phase -
                                            (classical_percentage*neg_phase_classical + quantum_percentage*neg_phase_quantum))

            self.weights = copy.deepcopy(new_weights)

            # Can add dump into files here









        # Maybe make cost and mixer Hamiltonians stored in self??
        # then can have circuit (qaoa) function within self and can access the Hamiltonians
        # Maybe include setting up hamiltonians in initialization?


    # Unclamped Samping step
    # Set up customized QAOA for unclamped sampling
    # def make_unclamped_QAOA(self):
    #
    #     # Indices
    #     visible_indices = [i for i in range(self.num_visible)]
    #     hidden_indices = [i + self.num_visible for i in range(self.num_hidden)]
    #
    #     # Setting up the full cost Hamiltonian
    #     coeffs_HC = np.array([])
    #     obs_HC = []
    #     for i in visible_indices:
    #         for j in hidden_indices:
    #             coeffs_HC = np.append(coeffs_HC, -1*self.weights[i][j-self.num_visible])
    #             obs_HC.append(qml.PauliZ(i) @ qml.PauliZ(j))
    #
    #     full_HC = qml.Hamiltonian(coeffs_HC, obs_HC)
    #
    #     # Setting up the full mixer Hamiltonian
    #     coeffs_HM = np.array([])
    #     obs_HM = []
    #     for i in (visible_indices + hidden_indices):
    #         coeffs_HM = np.append(coeffs_HM, 1)
    #         obs_HM.append(qml.PauliX(i))
    #
    #     full_HM = qml.Hamiltonian(coeffs_HM, obs_HM)

        # Build the cost and mixer layers
        # I am realizing now that I may need to break this into pieces differently













