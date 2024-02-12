#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
problem: crash on converting Qiskit MCX gate to tket
Solution: read QASM directly to TKet
'''


from qiskit import QuantumCircuit
from pytket.extensions.qiskit import qiskit_to_tk
from toolbox.activate_qtuum_api import activate_qtuum_api
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.qasm import circuit_from_qasm_str

nq=4
qc=QuantumCircuit(nq,1)
qc.h(0)
qc.ccx(0,1,2)
ctrL=[i for i in range(0,nq-1)]
qc.mcp(0.2,ctrL,nq-1)
qc.mcx(ctrL,nq-1)
qc.barrier()
qc.measure(0,0)
print(qc)
qasm_string=qc.qasm()

print('\n recovered from QASM')

qc1=circuit_from_qasm_str(qasm_string)
print('\n dump tket circ:')
for x in qc1:  print(x)

api=activate_qtuum_api()

backend = QuantinuumBackend(device_name='H1-1E', api_handler=api)
print('got backend',backend)

print('transpile ....')
qcT=backend.get_compiled_circuit(qc1, optimisation_level=2)
print('\n dump transpiled tket circ:')
for x in qcT:  print(x)
