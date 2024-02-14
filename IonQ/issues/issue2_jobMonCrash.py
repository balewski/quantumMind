#!/usr/bin/env python3
''' problem:  
 File "/usr/local/lib/python3.10/dist-packages/qiskit/tools/monitor/job_monitor.py", line 49, in _text_checker
    msg += " (%s)" % job.queue_position()
AttributeError: 'IonQJob' object has no attribute 'queue_position'


'''
from pprint import pprint
from time import time
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider
from qiskit.tools.monitor import job_monitor
import numpy as np


def par_bell_circ(nPar):
    qc = QuantumCircuit(nPar)
    ang=np.random.uniform(0, np.pi,nPar)
    for j in range(1,nPar):
        qc.rx(ang[j],j)
        qc.cx(0, j)
    qc.measure_all()
    return qc

# = = = = = = = = = = = =
#  M A I N 
# = = = = = = = = = = = =

shots=100000
qc=qc=par_bell_circ(10)
print(qc)

provider = IonQProvider()   # Remember to set env IONQ_API_KEY='....'
print(provider.backends())  # Show all backends
backend = provider.get_backend("ionq_simulator")

job = backend.run(qc, shots=shots, noise_model='harmony-2')
job_monitor(job)
counts = job.result().get_counts(0)
print(counts)
print('M:ok')

