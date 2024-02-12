#!/usr/bin/env python3

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from activate_azure import activate_azure_provider


# Create a quantum circuit acting on a single qubit
circuit = QuantumCircuit(1,1)
circuit.name = "Single qubit random"
circuit.h(0)
circuit.measure(0, 0)

# Print out the circuit
print(circuit)

# Create an object that represents Quantinuum's Syntax Checker target, "quantinuum.hqs-lt-s2-apival".
#   Note that any target you have enabled in this workspace can
#   be used here. Azure Quantum makes it extremely easy to submit
#   the same quantum program to different providers.

backend = activate_azure_provider("quantinuum.hqs-lt-s2-apival") # syntax checker

# Using the Quantinuum target, call "run" to submit the job. We'll
# use a count of 10 (simulated runs).
job = backend.run(circuit, count=10)
print("Job id:", job.id())

job_monitor(job)

result = job.result()

# The result object is native to the Qiskit package, so we can use Qiskit's tools to print the result as a histogram.
# For the syntax check, we expect to see all zeroes.
counts = result.get_counts()
print('counts=',counts)
print('M:done')

