#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Uses:
   provider = IBMProvider() 
   backend = provider.get_backend(args['backend'])
   qcT=transpile(...)
   job = backend.run(qcT,shots=..
   counts=result.get_counts(ic)
INPUT:
  nTask=args.get('num_task')
  shots=args.get('num_shot')
OUTPUT:
   just text w/ counts

'''

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, transpile

import quantum_serverless as servless

from qiskit_ibm_provider import IBMProvider
from time import time, sleep
from qiskit.providers.jobstatus import JobStatus
import numpy as np
import os
from qiskit.tools.visualization import circuit_drawer

import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num_shot',type=int,default=600, help="shots")
    parser.add_argument('-t','--num_task',type=int,default=2, help="tasks")
    parser.add_argument('-b','--backend',default="ibmq_qasm_simulator", help="tasks")    
    
    parser.add_argument("--out_path",default='out/',help="all outputs from  experiment")
    args = parser.parse_args()
    import hashlib
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    args.short_name='ghz_'+myHN
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    return vars(args)

#...!...!....................
def create_ghz_circuit(n):
    circuit = QuantumCircuit(n)
    circuit.h(0)
    for i in range(1, n):
        circuit.cx(0, i)
    circuit.measure_all()
    return circuit

#...!...!....................
def create_FeedF_circuit(n=4):
    qc = QuantumCircuit(n,n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    qc.barrier()
    qc.measure(1, 0)
    with qc.if_test((qc.cregs[0], 1)): qc.h(2)  # NEW
    #qc.h(2).c_if(qc.cregs[0], 1)  # OLD
    qc.measure(1, 1)
    return qc


#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":

    args = servless.get_arguments()    
    
    if len(args)==0:
        args=commandline_parser()
    else:  # exceptions:
        args['out_path']='/data'
         
    print('P:pattern3 short_name=%s start on %s'%(args['short_name'],args['backend']))
    print('M:IBMProvider()...')
    provider = IBMProvider()
    
    backend = provider.get_backend(args['backend'])
    print('\nmy backend=',backend)

    # from Aziz, patch backend for feed-forward 'NEW' logic
    #  with qc.if_test((qc.cregs[0], 1)): qc.h(2)  # NEW
    #  an issue opened on github https://github.com/Qiskit/qiskit-ibm-runtime/issues/1253. 
    if "if_else" not in backend.target:
            from qiskit.circuit import IfElseOp
            backend.target.add_instruction(IfElseOp, name="if_else")

    
    nTask=args['num_task']
    T0=time()
    T1=T0
    tarL=[]
    for it in range(nTask):
        print('\nP:pattern1 run task=%d ...'%it,backend)
        #qcE=create_ghz_circuit(it+2)
        qcE=create_FeedF_circuit()
        print('P:new circ, it:',it);print(qcE)
        qcT = transpile(qcE, backend=backend, optimization_level=3, seed_transpiler=42)
        print('P:transpiled to',backend)
        print(circuit_drawer(qcT.decompose(), output='text',cregbundle=True, idle_wires=False))

        job = backend.run(qcT,shots=args['num_shot'], dynamic=True)
        i=0
        while True:
            jstat=job.status()
            elaT=time()-T0
            print('P:i=%d task=%d status=%s, elaT=%.1f sec'%(i,it,jstat,elaT))
            if jstat==JobStatus.DONE: break
            i+=1; sleep(10)
        jstat=job.status()
        T2=time()
        elaT=T2-T0
        print('P: task=%d end-status=%s, elaT=%.1f sec,  taskDelay=%.1f sec'%(it,jstat,elaT,T2-T1))
        T1=T2
        if jstat==JobStatus.DONE :
            result = job.result()
            elaT=time()-T0
            ic=0
            counts=result.get_counts(ic)
            print('P: task=%d ended elaT=%.1f sec\n counts:'%(it,elaT), counts)
          
        else:
            print('P:job failed',jstat)
    MD={'my message':'all is good on %s'%str(backend)}

    servless.save_result(MD)
    print('P:pattern3 done',backend)
