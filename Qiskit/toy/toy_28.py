#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
example of desnity matrix computation using Qiskit 1.2



'''
from qiskit import  QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info.operators import Operator
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from pprint import pprint

# iSWAP matrix operator
iswap_op = Operator([[1, 0, 0, 0],
                     [0, 0, 1j, 0],
                     [0, 1j, 0, 0],
                     [0, 0, 0, 1]])
# Using operators in circuits
nq=2;nb=2
qc = QuantumCircuit(2)

# Add gates
qc.sdg(1)
qc.h(1)
qc.sdg(0)
qc.unitary(iswap_op, [0, 1], label='iswap')
qc.sdg(0)
qc.save_density_matrix()
qc.sdg(0)
qc.unitary(iswap_op, [0, 1], label='iswap')
qc.s(1)

print(qc)

backend1 = AerSimulator(method="density_matrix")
print('job started,  nq=%d  at %s ...'%(qc.num_qubits,backend1.name))

job = backend1.run(qc)
result = job.result() 
#!pprint(result)
print('ideal density matrix:',result.data())


print('\n repeat on  fake backend ...')
service = QiskitRuntimeService(channel="ibm_quantum")

backName='ibm_nazca'   # EPLG=3.2%

noisy_backend = service.backend(backName)
print('use noisy_backend =', noisy_backend.name )
backend2 = AerSimulator(method="density_matrix").from_backend(noisy_backend)


pm = generate_preset_pass_manager(optimization_level=1, backend=backend2)
qcT = pm.run(qc)
print('transpiled for',backend2)
print(qcT.draw('text', idle_wires=False))


job2 = backend2.run(qcT)
result2 = job2.result() 
print('noisy density matrix:',result2.data())

