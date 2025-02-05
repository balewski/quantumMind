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
    with qc.if_test((qc.cregs[0], 1)): qc.x(1)
    
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

    # ..... STEP 2 .....  run job
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit_aer import AerSimulator
    service = QiskitRuntimeService()
    
    if 1:
        backendN="ibm_torino" 
        backend = service.get_backend(backendN)
    else:
        print('run locally on Aer')
        backend = AerSimulator()  

    print('Transpile for',backend)
    qcT = transpile(qc3, backend=backend, optimization_level=3, seed_transpiler=111)
    sampler = Sampler(backend=backend)
    pub=(qcT,)
    print('job started at',backend)
    T0=time()
    job = sampler.run(pub, shots=4000)
    result=job.result()[0]
    elaT=time()-T0
    
    print('job ended elaT=%.1f sec\n probs:'%(elaT))
    
    counts=result.data.c.get_counts()      
    print('counts:%s',counts)
