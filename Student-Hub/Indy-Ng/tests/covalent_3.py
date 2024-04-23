import covalent as ct
from covalent_braket_plugin.braket import BraketExecutor
import os

# AWS resources to pass to the executor
credentials_file = "~/.aws/credentials"
profile = "default"
region = "us-east-1"
s3_bucket_name = "braket_s3_bucket"
ecr_repo_name = "braket_ecr_repo"
iam_role_name = "covalent-braket-iam-role"

# Instantiate the executor
ex = BraketExecutor(
            profile=profile,
            credentials=credentials_file,
            s3_bucket_name=s3_bucket_name,
            ecr_image_uri=ecr_image_uri,
            braket_job_execution_role_name=iam_role_name,
            quantum_device="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            classical_device="ml.m5.large",
            storage=30,
            time_limit=300,
    )


# Execute the following circuit:
# |0> - H - Measure
@ct.electron(executor=ex)
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