#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from qiskit import qpy
from qiskit.quantum_info import Operator

def show_circuit_parameters(qc, label=''):
    """Find and display all parameters in a quantum circuit"""
    params = qc.parameters
    print('M: circuit parameters%s:' % (' '+label if label else ''))
    print('  num_parameters=%d' % qc.num_parameters)
    if params:
        print('  parameter names:', [str(p) for p in params])
    else:
        print('  no parameters found')

def strip_measurements_get_unitary(qc):
    """Strip measurements from quantum circuit and return equivalent unitary
    
    Args:
        qc: Input quantum circuit
        
    Returns:
        numpy.ndarray: Unitary matrix of the circuit without measurements
    """
    # Create a copy of the circuit
    qc_copy = qc.copy()
    
    # Remove all measurement operations
    qc_copy.remove_final_measurements(inplace=True)
    
    # Convert to unitary operator
    unitary = Operator(qc_copy).data
    
    print('M: stripped measurements, unitary shape=%s' % str(unitary.shape))
    return unitary

inpF='qcrank_nqa3_nqd2.qpy'

with open(inpF, 'rb') as fd:
    qc = qpy.load(fd)[0]

n2q=qc.num_nonlocal_gates()
depth=qc.depth(filter_function=lambda x: x.operation.num_qubits == 2 )
print('\nCircuit   2q-gates=%d  depth=%d'%(n2q,depth))
print(qc.draw('text', idle_wires=False))

show_circuit_parameters(qc, 'original')

if 1:
    U=strip_measurements_get_unitary(qc)
    print('M: unitary shape=%s' % str(U.shape))
    print('M: unitary=',U)
    exit(1)

from iqm.qiskit_iqm import IQMProvider, transpile_to_IQM 

qpuName='sirius'
print('M: access IQM backend ...',qpuName)
provider=IQMProvider(url="https://cocos.resonance.meetiqm.com/"+qpuName)
backend = provider.get_backend()
print('got BCKN:',backend.name,qpuName)

qcT = transpile_to_IQM(qc, backend,optimization_level=3, seed_transpiler=43)
#print(qcT.draw('text', idle_wires=False))
physQubitLayout = qcT._layout.final_index_layout(filter_ancillas=True)
print('phys qubits:',physQubitLayout)

show_circuit_parameters(qcT, 'transpiled')
    
