#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Expectation value estimation using AER backend with Pauli observables
"""

from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime.options.estimator_options import EstimatorOptions
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator

import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int, help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-q','--num_qubit',type=int,default=3, help="qubit count")
    parser.add_argument('-n','--num_shot',type=int,default=2000, help="shots")
    
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def create_h_circuit(n):
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n, name="c")
    qc = QuantumCircuit(qr, cr)
    #qc.x(0) ; qc.x(1); return qc
    qc.h(0)
    for i in range(1, n):  qc.cx(0,i)
    qc.barrier()
    for i in range(0,n):  qc.measure(i,i)
    return qc

#=================================
#  M A I N
#=================================
if __name__ == "__main__":
    args=commandline_parser()
    #np.set_printoptions(precision=3)

    nq=args.num_qubit
    qc=create_h_circuit(nq)
    print(qc)

    obs = SparsePauliOp("I" * (nq-1)+"Z" ) # MSBF
    #obs = SparsePauliOp("Z"+"I" * (nq-1))  
    print('obs:',obs)

    backend = AerSimulator()
    print('job started,  nq=%d  at %s ...'%(qc.num_qubits,backend.name))
    options = EstimatorOptions()
    options.default_shots=args.num_shot
    estimator = Estimator(backend,options=options)

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    qcT = pm.run(qc)
    obsT=obs.apply_layout(qcT.layout)

    #job = estimator.run([(qc,obs)])  # works too
    job = estimator.run([(qcT,obsT)])
    
    result = job.result()
    rdata=result[0].data
    print("Expectation value: %.3f +/- %.3f "%(rdata.evs,rdata.stds))
    print("Metadata: ",result[0].metadata)

