#!/usr/bin/env python3

from qiskit_ionq import IonQProvider
from qiskit import QuantumCircuit
from time import sleep
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


qc_bell = QuantumCircuit(2, name=" example with noise")
qc_bell.h(0)
qc_bell.cx(0, 1)
qc_bell.measure_all()

# Create the provider (make sure your API key is set up)
provider = IonQProvider()

if 1:  # simu
    qc_bell.name='simu forte-1 '+  qc_bell.name
    backend = provider.get_backend("simulator")
    backend.set_options(noise_model="forte-1")
    print('runs on simulator')
else:
    qc_bell.name='real forte-1 '+  qc_bell.name
    backend = provider.get_backend("qpu.forte-1")
    print('runs on real QPU')
    
if 1:
    job= backend.run(qc_bell, shots=200)
    print('use backend.run()')
else:
    sampler = Sampler(mode=backend)
    print('use sampler.run()')
    job =  sampler.run([qc_bell], shots=100)

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

