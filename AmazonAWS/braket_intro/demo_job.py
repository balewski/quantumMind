from braket.jobs.metrics import log_metric
from braket.jobs import hybrid_job
from braket.jobs.environment_variables import get_job_device_arn
from braket.tracking import Tracker
from braket.devices import Devices

import pennylane as qml
from pennylane import numpy as np


def create():
    def qubit_rotation(device, num_steps, stepsize):

        @qml.qnode(device)
        def circuit(params):
            qml.RX(params, wires=0)
            return qml.expval(qml.PauliZ(0))

        opt = qml.GradientDescentOptimizer(stepsize=stepsize)
        params = np.array([0.1])

        for i in range(num_steps):
            params = opt.step(circuit, params)
            expval = circuit(params)

            log_metric(metric_name="theta", iteration_number=i, value=float(params[0]))
            log_metric(metric_name="expval", iteration_number=i, value=expval)

        return params.tolist()[0]


    @hybrid_job(device=Devices.IQM.Garnet)
    def qubit_rotation_qpu_job(num_steps=15, stepsize=0.5, shots=1000):
        cost_tracker = Tracker().start()

        device = qml.device("braket.aws.qubit", device_arn=get_job_device_arn(), wires=1, shots=shots)
        optimal_theta = qubit_rotation(device, num_steps=num_steps, stepsize=stepsize)

        cost_tracker.stop()
        return {
            "theta": optimal_theta,
            "task summary": cost_tracker.quantum_tasks_statistics(),
            "estimated cost": float(cost_tracker.simulator_tasks_cost() + cost_tracker.qpu_tasks_cost()),
        }

    job = qubit_rotation_qpu_job()
    return job