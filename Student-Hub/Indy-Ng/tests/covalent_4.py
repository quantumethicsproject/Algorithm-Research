import covalent as ct
import os

# Statevector simulator
sv1 = ct.executor.BraketQubitExecutor(
    device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    shots=1024,
    s3_destination_folder=(),
)


# Execute the following circuit:
# |0> - H - Measure
@ct.electron
def simple_quantum_task(num_qubits: int):
    import pennylane as qml

    # These are passed to the Hybrid Jobs container at runtime
    device_arn = os.environ["AMZN_BRAKET_DEVICE_ARN"]
    s3_bucket = os.environ["AMZN_BRAKET_OUT_S3_BUCKET"]
    s3_task_dir = os.environ["AMZN_BRAKET_TASK_RESULTS_S3_URI"].split(s3_bucket)[1]

    device = qml.device(
        "braket.aws.qubit",
        device_arn=device_arn,
        s3_destination_folder=(s3_bucket, s3_task_dir),
        wires=num_qubits,
    )

    @ct.qelectron(executor=sv1)
    @qml.qnode(device=device)
    def simple_circuit():
        qml.Hadamard(wires=[0])
        return qml.expval(qml.PauliZ(wires=[0]))

    res = simple_circuit().numpy()
    return res


@ct.lattice
def simple_quantum_workflow(num_qubits: int):
    return simple_quantum_task(num_qubits=num_qubits)


dispatch_id = ct.dispatch(simple_quantum_workflow)(1)
result_object = ct.get_result(dispatch_id, wait=True)

# We expect 0 as the result
print("Result:", result_object.result)