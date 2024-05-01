#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options, Session
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.visualization import circuit_drawer
   

from time import time, sleep
from qiskit.providers.jobstatus import JobStatus
import numpy as np
import h5py,os

import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num_shot',type=int,default=1000, help="shots")
    parser.add_argument('-b','--backend',default="ibm_torino", help="tasks")
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))   
    return vars(args)

#...!...!....................
def circ_2q_nonlin(alpha,theta):
    # Create a Quantum Circuit with 2 qubits and 2 classical bits for measurement
    qc = QuantumCircuit(2, 2)
    qc.rx(alpha, 0)
    qc.ry(theta, 1)
    qc.cx(1, 0)
    qc.measure(0, 0)  
    # Add a conditional H gate on qubit 1 depending on the classical bit 0
    with qc.if_test((qc.cregs[0], 1)): qc.h(1)  # NEW

    # Measure qubit 1 into classical bit 1
    qc.measure(1, 1)
    return qc

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=commandline_parser()
    backendN=args['backend']
    shots=args['num_shot']
    
    print('P:pattern1 start on',backendN)

    service = QiskitRuntimeService()
 
    # Run on a simulator
    backend = service.get_backend(backendN)
 
    options = Options()
    options.optimization_level=3  
    options.resilience_level = 1
    options.execution.shots =shots
    session = Session(backend=backend)
    sampler = Sampler(session=session, options=options)

    T0=time()
    qcP=circ_2q_nonlin(0.3, 0.7)
    
    print(circuit_drawer(qcP, output='text',cregbundle=False))
    qcL=[qcP]
    print('P:run %d circuits ...'%len(qcL))
    job = sampler.run(qcL)
    elaT=time()-T0
    result = job.result()
    print('P:  ended elaT=%.1f sec\n probs:'%(elaT))
    nCirc=len(qcL)
    for ic in range(nCirc):
        probD=result.quasi_dists[ic]
        print('ic=%d probD:%s'%(ic,probD))

    print('P:pattern1 done, shots:',shots,backend)
