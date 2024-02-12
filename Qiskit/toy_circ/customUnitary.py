#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
example from 
Qiskit Aer: Applying noise to custom unitary gates

'''
from qiskit import execute, QuantumCircuit, QuantumRegister, ClassicalRegister, BasicAer,transpile
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import process_fidelity
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel, errors

# fix it
#IBMQ.load_accounts()

# Creating matrix operators
# CNOT matrix operator with qubit-0 as control and qubit-1 as target
cx_op = Operator([[1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0]])

# iSWAP matrix operator
iswap_op = Operator([[1, 0, 0, 0],
                     [0, 0, 1j, 0],
                     [0, 1j, 0, 0],
                     [0, 0, 0, 1]])
# Using operators in circuits
nq=2;nb=2
cx_circ = QuantumCircuit(2)

# Add gates
cx_circ.sdg(1)
cx_circ.h(1)
cx_circ.sdg(0)
cx_circ.unitary(iswap_op, [0, 1], label='iswap')
cx_circ.sdg(0)
cx_circ.h(0)
cx_circ.sdg(0)
cx_circ.unitary(iswap_op, [0, 1], label='iswap')
cx_circ.s(1)

meas = QuantumCircuit(nq, nb)
#meas.barrier(range(nq))
meas.measure(range(nq), range(nb))

meas.compose(cx_circ, inplace=True)


# execute the quantum circuit 
backend = BasicAer.get_backend('qasm_simulator') # the device to run on
result = execute(meas, backend, shots=1000).result()
countsD  = result.get_counts(meas)
print('raw counts',countsD)
print(meas)

from compare_circ_unitaries import  circ_depth_aziz
circ_depth_aziz(meas,'ideal')

basis_gates = ['p','u3','cx'] 
qc1 = transpile(meas, backend, basis_gates= basis_gates,optimization_level=3)
circ_depth_aziz(qc1,'transp')
print(qc1)



