#!/usr/bin/env python3

from qiskit_ionq import IonQProvider
from qiskit import QuantumCircuit
from time import sleep
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


qc_bell = QuantumCircuit(2, name="Sim example with noise")
qc_bell.h(0)
qc_bell.cx(0, 1)
qc_bell.measure_all()

# Create the provider (make sure your API key is set up)
provider = IonQProvider()
backend = provider.get_backend("simulator")
backend.set_options(noise_model="forte-1")

if 0:
    job= backend.run(qc_bell, shots=1000)
    print('use backend.run()')
else:
    sampler = Sampler(mode=backend)
    print('use sampler.run()')
    job =  sampler.run([qc_bell], shots=1000)

jid=job.job_id()
print('full jobID:',jid, job.status())
sleep(10)
print('full jobID:',jid, job.status())

# Retrieve the job by ID
job = backend.retrieve_job(jid)

print("Job status:", job.status())
result = job.result()
counts = result.get_counts()

print("Final counts:", counts)

