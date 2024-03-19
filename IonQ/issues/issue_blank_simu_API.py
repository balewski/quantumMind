#!/usr/bin/env python3
''' problem: ??

'''
from pprint import pprint
from time import time
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider
# Setup noise model as one of:
simList=['ideal', 'harmony-2', 'aria-1', 'aria-2', 'forte-1']


def bell_circ():
    qc = QuantumCircuit(2,2)
    qc.h(0)
    qc.cx(0, 1) 
    qc.measure_all()
    return qc

# = = = = = = = = = = = =
#  M A I N 
# = = = = = = = = = = = =

shots=1000
qc=bell_circ()
print(qc)

provider = IonQProvider()   # Remember to set env IONQ_API_KEY='....'
print(provider.backends())  # Show all backends
backend = provider.get_backend("ionq_simulator")

job = backend.run(qc, shots=shots) #, noise_model=noiseN)
counts = job.result().get_counts(0)
print(counts)
print('M:ok')

