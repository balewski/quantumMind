#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Random circuit generation and execution with density matrix simulation

Generate random large circuit and run it using density matrix simulator which is very slow
"""

from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister,transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from time import time

import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int, help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-q','--num_qubit',type=int,default=3, help="qubit count")
    parser.add_argument('-n','--num_shot',type=int,default=2000, help="shots")
    parser.add_argument( "-R","--addReverse",  action='store_true', default=False, help="add inverted unitary so circuit returns 0^nq state only")
    
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def generate_random_circuit(nq=5, addRev=False, depth=5):
    """
    Generates a random quantum circuit on nq qubits.
    
    Parameters:
        nq (int): Number of qubits.
        addRevers (bool): 
            - If False, returns a random circuit with measurements.
            - If True, returns a circuit composed of a random circuit (without measurement),
              its inverse, and then adds measurements on all qubits.
        depth (int): Depth of the random circuit.
        
    Returns:
        QuantumCircuit: The generated quantum circuit.
    """
    basis_gates = ['u3', 'cx']
    qc1 = random_circuit(nq, depth=depth, measure=False)
    qc = transpile(qc1, backend=backend, optimization_level=1, basis_gates=basis_gates)
    qc_inv = qc.inverse()
    if addRev:
        qc.barrier()
        qc=qc.compose(qc_inv)
        
    qc.measure_all()
    return qc



#=================================
#  M A I N
#=================================
if __name__ == "__main__":
    args=commandline_parser()
   
    backend = AerSimulator()    
    nq=args.num_qubit
    qc=generate_random_circuit(nq,args.addReverse)
    if nq<4: print(qc)

    
    print('job started,  nq=%d  at %s ...'%(qc.num_qubits,backend.name))
    options = SamplerOptions()
    options.default_shots=10000

    qcEL=(qc,)  # quant circ executable list
    sampler = Sampler(mode=backend, options=options)
    T0=time()
    job = sampler.run(qcEL)
    result=job.result()
    elaT=time()-T0
    counts=result[0].data.meas.get_counts()
    print('counts:',counts)
    print('run time %.1f sec'%(elaT))
