#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Uses:
  estimator = Estimator(backend)
  job=estimator.run(circuit, observable)
  EVs = job.result().values.tolist()
'''

from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator
from quantum_serverless import save_result
 
from time import time, sleep
from qiskit.providers.jobstatus import JobStatus

print('P:pattern2 start')
service = QiskitRuntimeService()
 
# Run on a simulator
backend = service.get_backend("ibmq_qasm_simulator")
#backend = service.get_backend("ibmq_kolkata")
#backend = service.get_backend("ibmq_mumbai")
#backend = service.get_backend("ibm_algiers")
#backend = service.get_backend("ibm_hanoi")

circuit = random_circuit(2, 2, seed=1234)
print('P:circ');print(circuit)
observable = SparsePauliOp("IY")
 
estimator = Estimator(backend)
nTask=3
T0=time()
for it in range(nTask):
    print('\nP:pattern2 run task=%d ...'%it,backend)

    job = estimator.run(circuit, observable)
    for _ in range(100):
        jstat=job.status()
        elaT=time()-T0
        print('P: task=%d status=%s, elaT=%.1f sec'%(it,jstat,elaT))
        if jstat==JobStatus.DONE: break
        sleep(10)
    jstat=job.status()
    elaT=time()-T0
    print('P:end-status=%s, elaT=%.1f sec'%(jstat,elaT))

    if jstat==JobStatus.DONE :
        result = job.result()
        elaT=time()-T0
        print('P: task=%d ended elaT=%.1f sec\n Expectation values:'%(it,elaT), result.values.tolist())
    else:
        print('P:job failed',jstat)
        
print('P:pattern2 done')
