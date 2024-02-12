#!/usr/bin/env python3
''' problem:  what is transpiled depth of circuit for each HW and and what is the  cost of execution on the HW 

'''
from pprint import pprint
from time import time
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider
# Setup noise model as one of:
simList=['ideal', 'harmony-2', 'aria-1', 'aria-2', 'forte-1']


def par_bell_circ(nPar):
    n2=2
    qc = QuantumCircuit(n2*nPar)
    for j in range(nPar):
        i0=j*n2; i1=i0+1
        qc.h(i0)
        qc.cx(i0, i1)
    qc.measure_all()
    return qc

# = = = = = = = = = = = =
#  M A I N 
# = = = = = = = = = = = =
nPar=6
shots=1000
qc=par_bell_circ(nPar)
print(qc)

provider = IonQProvider()   # Remember to set env IONQ_API_KEY='....'
print(provider.backends())  # Show all backends
backend = provider.get_backend("ionq_simulator")

import random
for noiseN in simList:
    #noiseN='harmony-1'
    print('\nIonQ simu: %s nQ=%d ....'%(noiseN,qc.num_qubits),end='')
    T0=time()
    try:
        job = backend.run(qc, shots=shots, noise_model=noiseN)
        counts=job.get_counts()
    except:
        print("Failed"); continue
    
    elaT=time()-T0
    # Find the bitstring with the maximum count
    max_bitstring = max(counts, key=counts.get)
    print('elaT=%.1f sec, num bistrings=%d,  max mshot=%d for %s'%(elaT,len(counts),counts[max_bitstring],max_bitstring))
    print('circ depth XXX, cost on HW YYY')
    random_samples = random.sample(list(counts.items()), 5)
    for bitstring, count in random_samples: # Print the selected random samples
        print("   %s:%d,"%(bitstring,count),end='')
    print()
   
print('M:ok')

