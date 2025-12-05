#!/usr/bin/env python3

from qiskit_ionq import IonQProvider, ErrorMitigation
from qiskit import QuantumCircuit, transpile
from time import sleep
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# Print all enum names and values
for em in ErrorMitigation:
    print(em.name, em.value)

    
qc = QuantumCircuit(2, name="my bell")
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Create the provider (make sure your API key is set up)
provider = IonQProvider()

if 1:  # simu
    qc.name='simu forte-1 '+  qc.name
    backend = provider.get_backend("simulator")
    backend.set_options(noise_model="forte-1")
    print('runs on simulator')
else:
    qc.name='real forte-1 '+  qc.name
    backend = provider.get_backend("qpu.forte-1")
    print('runs on real QPU',backend.name)

print(qc)

qcT=transpile(qc, backend=backend, optimization_level=1)

print(qcT)

errMit=ErrorMitigation.NO_DEBIASING
errMit=ErrorMitigation.DEBIASING
shots=902

if 1:
    print('use backend.run()',errMit)        
    job= backend.run(qcT, shots=shots,error_mitigation=errMit)
else:
    not_working
    sampler = Sampler(mode=backend)
    print('use sampler.run()')
    job =  sampler.run([qc_bell], shots=shots)

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

#provider = IonQProvider()
#backend_native = provider.get_backend("simulator", gateset="native")
#qc_native = transpile(qc_abstract, backend=backend_native)
#qc.draw()
