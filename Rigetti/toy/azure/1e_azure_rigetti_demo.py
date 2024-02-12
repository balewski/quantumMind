#!/usr/bin/env python3

from activate_azure import activate_azure_provider

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

# Create an object that represents Rigetti sim 
#quantinuum_api_val_backend = provider.get_backend("rigetti.qpu.aspen-11") # Rigetti HW

# Create a quantum circuit acting on a single qubit
circuit = QuantumCircuit(1,1)
circuit.name = "Single qubit random"
circuit.h(0)
circuit.measure(0, 0)

# Print out the circuit
print(circuit)

shots=10
job = backend.run(circuit, count=shots);  jobId= job.id(); print("Job id:",jobId)
#jobId='4a0aaa00-49e6-11ed-8d57-0242ac110003'
job = backend.retrieve_job(jobId)

job_monitor(job)

result = job.result()

# The result object is native to the Qiskit package, so we can use Qiskit's tools to print the result as a histogram.
# For the syntax check, we expect to see all zeroes.
counts = result.get_counts()
print('counts=',counts)
print('M:done')

