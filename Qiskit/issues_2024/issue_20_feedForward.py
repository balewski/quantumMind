#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from qiskit import QuantumCircuit, transpile
from qiskit.visualization import circuit_drawer
from time import time
import numpy as np

#...!...!....................                
def input(nq):    
    qc = QuantumCircuit(nq,name='feature_circ')
    for i in range(nq): qc.rx(np.pi/2,i)
    return qc

#...!...!....................
def sum2brick():
    qc = QuantumCircuit(2,1,name='ansatz_circ')
    qc.cx(1,0)
    qc.measure(1, 0)        
    with qc.if_test((qc.cregs[0], 1)): qc.x(1)  # NEW
    #qc.x(1).c_if(qc.cregs[0], 1)  # OLD
    return qc
    
#=================================
#  M A I N
#=================================
#=================================

if __name__ == "__main__":

    # ..... STEP 1 .....  construct 
    nq=3
    qc1=input(nq)
    #1print(circuit_drawer(qc1, output='text',cregbundle=False))

    qc2=sum2brick()
    #1print(circuit_drawer(qc2, output='text',cregbundle=False))
    #print(qc2)  # WORKS
    
    qc3= QuantumCircuit(nq,1)
    qc3.append(qc1, range(nq))               
    qc3.compose(qc2, [1,2], [0], inplace=True)   
    qc3.barrier()
    qc3.measure(2,0)
    
    print('use circuit_drawer:')
    print(circuit_drawer(qc3.decompose(), output='text',cregbundle=False))
    
    # print('use str()') ;    print(qc3)  # CRASH1 - low priority

    # ..... STEP 2 .....  run job
    #backendN="ibmq_qasm_simulator"    
    backendN="ibm_cusco"    
    backendN="ibm_torino"    

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    backend = service.get_backend(backendN)

    print('Transpile for',backend)
    qcT = transpile(qc3, backend=backend, optimization_level=3, seed_transpiler=111)
   
    print('job started at',backend)
    T0=time()
    job=backend.run(qcT,shots=4000, dynamic=True)
    result=job.result()
    elaT=time()-T0
   
    print('job ended elaT=%.1f sec\n probs:'%(elaT))
    
    counts=result.get_counts(0)       
    print('counts:%s',counts)
