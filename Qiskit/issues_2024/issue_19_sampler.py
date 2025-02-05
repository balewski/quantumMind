#!/usr/bin/env python3
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options, Session

#backName = "ibm_cairo"
backName='ibmq_qasm_simulator'

service = QiskitRuntimeService()
backend = service.get_backend(backName)
print('M: backend=',backend)
session = Session(backend=backend)
options = Options()
options.resilience_level = 0
options.execution.shots =1024
sampler = Sampler(session=session, options=options)
print('M:sampler OK, start job ...')

from qiskit.circuit.random import random_circuit
circuit = random_circuit(2, 2, seed=0, measure=True)
job = sampler.run(circuit)
result = job.result()
print(result)

# recover shots
ic=0 # circuit index
tshots=result.metadata[ic]['shots']
qdist=result.quasi_dists[ic]
for k in qdist:
    p=qdist[k]
    m=p*tshots
    print('k=%d p=%.3f, mshot=%.1f'%(k,p,m))
print('M:ok')

