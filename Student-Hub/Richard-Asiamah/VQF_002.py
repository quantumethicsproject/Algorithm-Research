
import pennylane as qml
from pennylane import numpy as np

# Specify the number to be factored
N = 35

# Specify the number of qubits and number of iterations
n_qubits = 2
n_iterations = 100

# Specify the backend and device
dev = qml.device("default.qubit", wires = n_qubits)

# Define the quantum circuit
def circuit(x, wires):
  # Apply Hadamard gate to all wires
  qml.broadcast(unitary=qml.Hadamard, pattern = "single", wires = dev.wires)
  
  # Apply U function
  for i in range(len(wires) - 1):
        qml.CRZ(x[i], wires = [wires[i], wires[i + 1]])
  qml.CRZ(x[-1], wires = [ wires[-1], wires[0]])
  
  return [qml.expval(qml.PauliZ(wires = wire)) for wire in wires]

# Define the cost function
def cost(x):
  #Evaluate the quantum circuit
  result = circuit(x, wires = list(range(n_qubits)))
  
  # Extract the observables from the result
  observables = [qml.PauliZ(wire) for wire in range(n_qubits)]
  expectation_values = [qml.expval(obs(result)) for obs in observables]
  
  return np.abs(expectation_values[0] - expectation_values[-1])

# Define the optimization loop
def optimize_circuit():
  # Initialize the optimization parameters
  opt = qml.GradientDescentOptimizer(stepsize = 0.4)
  x = np.random.uniform(0, 2*np.pi, size = n_qubits)
  
  # Optimization loop
  for i in range(n_iterations):
    x = opt.step(cost, x)
    if (i + 1) % 10 == 0:
      printf(f"Cost after iteration {i + 1}: { cost(x)}")
      
   return x

# Run the optimization
x_opt = optimize_circuit()

# produce the factors
factors = np.gcd(int(2**(n_qubits)/2) + int(np.round(np.sin(x_opt[0])**2 * 2**(n_qubits)/2)), N) np.gcd(int(2**(n_qubits)/2) - int(np.round(np.sin(x_opt[0])**2 * 2**(n_qubits/2)), N) 
  
  
  
  
