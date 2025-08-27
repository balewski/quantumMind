#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
from qiskit import qpy
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler 
from qiskit import transpile


inpF='inc_advection_solver_set400.qpy'

with open(inpF, 'rb') as fd:
    qcL = qpy.load(fd)

# assemble circuit
nq=qcL[0].num_qubits
print('nq=',nq)
qc=QuantumCircuit(nq)
for i in range(2):
    #if i==0: continue
    qc.append(qcL[i],range(nq))
    
qc.measure_all()
print(qc)

print('M: activate QiskitRuntimeService() ...')
service = QiskitRuntimeService()

backN='ibm_kingston'
backend = service.backend(backN , use_fractional_gates=True)  #  enable rzz gates
print('use true HW backend =', backN) 

qcT0 =  transpile(qc, backend,optimization_level=3)
qcT =  transpile(qcT0, backend,optimization_level=3)
len2q=qcT.depth(filter_function=lambda x: x.operation.num_qubits == 2 )
n2q_g=qcT.num_nonlocal_gates()
print('Transpiled, ops:',qcT.count_ops(),'\nnum2q:',n2q_g,'len2q:',len2q)
