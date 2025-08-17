#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
from qiskit import qpy
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler 
from qiskit import transpile


coreN='advection_solver_set1'
inpF='../ibmq_data/inc_advection_solver_aug16/ibmq_qc_inc_%s.qpy'%coreN


with open(inpF, 'rb') as fd:
    qcL = qpy.load(fd)

# assemble circuit
nq=qcL[0].num_qubits
print('nq=',nq)
qc=QuantumCircuit(nq)
for i in range(2):
    qc.append(qcL[i],range(nq))
qc.measure_all()
print(qc)

print('M: activate QiskitRuntimeService() ...')
service = QiskitRuntimeService()

backN='ibm_kingston'
backend = service.backend(backN , use_fractional_gates=True)  #  enable rzz gates
print('use true HW backend =', backN) 

qcT =  transpile(qc, backend,optimization_level=3)
print('Transpiled, ops:',qcT.count_ops())

sampler = Sampler(mode=backend)
job = sampler.run(tuple([qcT]),shots=100)

print('circ submitted')
