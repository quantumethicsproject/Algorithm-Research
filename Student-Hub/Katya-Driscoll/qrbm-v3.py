
"""
DESCRIPTION
"""

# Modules
import copy
import pennylane as qml
from pennylane import numpy as np
from pennylane import qaoa

"""
function description goes here
"""

device = qml.device('default.qubit', wires=5)

class qRBM:
    """
    Fields:         _dev                = the quantum device being used
                    _n_visible          = number of visible nodes
                    _n_hidden           = number of hidden nodes
                    _n_total            = total number of nodes (hidden and visible)
                    _n_qaoa_steps       = desired number of qaoa steps
                    _beta_temp          = temperature of the system, angle beta for preparing thermal state
                    _use_bias           = T/F whether bias is being used or not
                    _state_prep_angle   = used in the initial circuit
                    _veq_inst           = ???
                    _weights            = the weights. weights[i][j] = weight between node(i) and node(j)
                    _biases             = the biases for each node
                    _full_HC            = the current full cost Hamiltonian
                    _full_HM            = the current full mixer Hamiltonian
                    _optimizer          = a classical optimizer for part of the process of updating gamma, nu
    """

    def __init__(self, device, num_visible_nodes, num_hidden_nodes, qaoa_steps=1, beta_temp=1.0, useBias=False):
        """
        Constructor

        """
        self._dev = device
        self._n_visible = num_visible_nodes
        self._n_hidden = num_hidden_nodes
        self._n_total = self._n_visible + self._n_hidden
        self._n_qaoa_steps = qaoa_steps
        self._beta_temp = beta_temp
        self._use_bias = useBias

        # State angle used in initial circuit for qaoa
        # Check this?
        self._state_prep_angle = np.arctan(np.exp(-1/self._beta_temp)) * 2.0

        # Classical optimizer used to update parameters after QAOA
        self._optimizer = qml.GradientDescentOptimizer()

        # VQE thing here

        # Initialize the weights to something random
        # Using a random uniform distribution in this case
        self._param_wb = 0.1*np.sqrt(6 / self._n_total)         # Parameter for initializing weights and biases
        self._weights = np.asarray(
            np.random.uniform(
                low=-self._param_wb, high=self._param_wb,
                size=(self._n_visible, self._n_hidden)
            )
        )

        # Initialize the biases to something random
        if useBias:
            self._biases = np.asarray(
                np.random.uniform(
                    low=-self._param_wb, high=self._param_wb,
                    size=(self._n_hidden)
                )
            )
        else:
            self._biases = None

        # Initialize the cost and mixer Hamiltonians
        self._full_HC, self._full_HM = self.update_Hamiltonians()


    def update_Hamiltonians(self):
        """
        To set up/update the cost and mixer Hamiltonians

        update_Hamiltonians: None -> None
        """

        # Indices for loops
        vis_indices = [i for i in range(num_visible_nodes)]
        hid_indices = [i + num_visible_nodes for i in range(num_hidden_nodes)]
        tot_indices = [i for i in range(num_total_nodes)]
        anc_indices = [i + num_total_nodes for i in range(num_ancillaries)]
    
        # Full cost Hamiltonian
        coeffs_HC = np.array([])
        obs_HC = []
        for i in vis_indices:
            for j in hid_indices:
                coeffs_HC = np.append(coeffs_HC, 1 * weights[i][j - num_visible_nodes])
                obs_HC.append(qml.PauliZ(i) @ qml.PauliZ(j))
        full_HC = qml.Hamiltonian(coeffs_HC, obs_HC)
    
        # Full mixer Hamiltonian
        coeffs_HM = np.array([])
        obs_HM = []
        for i in tot_indices:
            coeffs_HM = np.append(coeffs_HM, 1.0)
            obs_HM.append(qml.PauliX(i))
        full_HM = qml.Hamiltonian(coeffs_HM, obs_HM)

        return full_HC, full_HM


    def unclamped_qaoa_layer(self, gamma, nu):         # params[0] = gamma, params[1] = nu
        """
        One layer of QAOA for the unclamped sampling process

        unclamped_qaoa_layer: [gamma, nu] -> None
        """
        # Pennylane does this for you, nice
        qaoa.cost_layer(gamma, HC)
        qaoa.mixer_layer(nu, HM)


    def unclamped_qaoa_circuit(self, params):
        """
        The (layered) QAOA circuit for the unclamped sampling process

        unclamped_qaoa_circuit: [gamma, nu] -> None
        """
        # Initializing the circuit
        for node in range(num_total_nodes):
            j = node + num_total_nodes                   # ANCILLARY QUBITS
            # RX on that node
            qml.RX(phi=angle, wires=j)   
            # entangle hidden and visible nodes??
            qml.CNOT([j, node])
        
        qml.layer(unclamped_qaoa_layer, depth=2, gamma=params[0], nu=params[1])

    @qml.qnode(device)
    def unclamped_cost_function(self, params):

        """
        Measure the expected value of the cost function after applying the unclamped qaoa circuit

        unclamped_cost_function: [gamma, nu] -> Float
        """
        self.unclamped_qaoa_circuit(params)
        return qml.expval(self._full_HC)

    def unclamped_sampling(self):
        """
        Unclamped sampling/thermalization process

        unclamped_sampling: None -> [gamma, nu]
        """

        # Initialize gamma, nu as random
        # For now setting to 0.5 each
        gamma = 0.5
        nu = 0.5

        # Optimize the cost function circuit
        op_steps = 70
        params = np.array(params, requires_grad=True)
    
        optimizer = qml.GradientDescentOptimizer()
        for i in range(op_steps):
            params = optimizer.step(unclamped_cost_function, params)
        
        return params


    @qml.qnode(device)
    def unclamped_2sz(self, gamma, nu, i, j):
        """
        Unclamped sampling: measuring expectation values to update weights/biases

        :param gamma:
        :param nu:
        :param i:
        :param j:
        :return:
        """
        self.unclamped_qaoa_circuit([gamma, nu])
        return qml.PauliZ(i) * qml.PauliZ(j)


    @qml.qnode(device)
    def unclamped_sz(self, gamma, nu, i):
        """
        Unclamped sampling: measuring expectation value to update biases

        :param gamma:
        :param nu:
        :param i:
        :return:
        """
        self.unclamped_qaoa_circuit([gamma, nu])
        return qml.PauliZ(i)


    def sigmoid(self, num):
        """
        Helper function to calculate sigmoid function of number.
        Note the sigmoid function maps all numbers to something in range[0,1] (turns everything to probabilities)

        sigmoid: Float -> Float
        """

        return 1.0 / (1.0 + np.exp(-num))


    def train(self, data, learning_rate=0.1, num_epochs=100, quantum_percentage=1.0, classical_percentage=0.0):
        """
        Trains the RBM

        :param data:
        :param learning_rate:
        :param num_epochs:
        :param quantum_percentage:
        :param classical_percentage:
        :return:
        """

        # NEED TO ADD NUM QUANTUM MEASUREMENTS


        # NOTE: must have quantum_percentage + classical_percentage == 1.0
        assert quantum_percentage + classical_percentage == 1.0

        # Convert data format
        data = np.asarray(data)

        # Train over chosen number of epochs
        for epoch in range(num_epochs):
            # get optimal parameters
            [gamma, nu] = self.unclamped_sampling()

            # Allocating space for updates
            unc_neg_phase_quantum_weights = np.zeros_like(self._weights)
            # measure expvals sZ using optimal parameters
            for i in range(self._n_visible):
                for j in range(self._n_hidden):
                    unc_neg_phase_quantum_weights[i][j] =  self.unclamped_2sz(gamma, nu, i, j)
            unc_neg_phase_quantum_weights *= (1. / float(len(data)))

            # Update biases if needed
            if self._biases is not None:
                unc_neg_phase_quantum_biases = np.zeros_like(self._biases)
                for i in range(self._n_hidden):
                    unc_neg_phase_quantum_biases[i] = self.unclamped_sz(gamma, nu, i)
                unc_neg_phase_quantum_biases *= (1. / float(len(data)))

            # Determine which nodes will be activated
            # need to add bias here
            pos_hidden_probs = self.sigmoid(np.dot(data, self._weights))
            pos_hidden_states = pos_hidden_probs > np.random.rand(len(data), self._n_hidden)
            pos_phase_classical = np.dot(data.T, pos_hidden_probs) * (1. / float(len(data)))

            neg_visible_activations = np.dot(pos_hidden_states, self._weights.T)
            neg_visible_probs = self.sigmoid(neg_visible_activations)

            neg_hidden_activations = np.dot(neg_visible_probs, self._weights)
            neg_hidden_probs = self.sigmoid(neg_hidden_activations)


            neg_phase_classical = np.dot(neg_visible_probs.T, neg_hidden_probs) * 1./len(data)

            # Get new weights
            new_weights += learning_rate * (pos_phase_classical - \
                          (classical_percentage * neg_phase_classical + \
                           quantum_percentage * unc_neg_phase_quantum_weights))

            # Get new biases


            # Update weights and biases
            self._weights = copy.deepcopy(new_weights)
            # bias

        # End of training
        print("Training Done!")







