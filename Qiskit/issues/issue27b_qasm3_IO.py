#!/usr/bin/env python3
''' problem:  
When a simple circuit,  like GHZ, is transpiled  for a backend, next saved as Qasm3,
the qiskit.qasm3.loads( .)  forgets the qubit mapping assigned by the transpiler and counts qubits from 0 to N-1. Such a read-in  circuit  would not run on the HW properly.


'''
from pprint import pprint
from qiskit import   QuantumCircuit, transpile

from qiskit_ibm_runtime import QiskitRuntimeService
import qiskit.qasm3

#...!...!....................
def create_ghz_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):  qc.cx(0, i)
    qc.measure_all()
    return qc


#=================================
if __name__ == "__main__":

    qc=create_ghz_circuit(4) 

    print(qc)

    backName = "ibm_torino"  #CZ

    service = QiskitRuntimeService()
    backend = service.get_backend(backName)
    print('M: backend=',backend)
    if "if_else" not in backend.target:
        from qiskit.circuit import IfElseOp
        backend.target.add_instruction(IfElseOp, name="if_else")

    print('\n transpile\n')
    qcT = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=111)
    print(qcT.draw(output='text',idle_wires=False,cregbundle=False))  # skip ancilla

    print('M: export QASM3  circ:\n')
    qasm1=qiskit.qasm3.dumps(qcT)
    print('\nM: parse Qasm3')
    qc2=qiskit.qasm3.loads(qasm1)
    print('M: print imported qc2')
    print(qc2.draw(output='text',idle_wires=False,cregbundle=False)) 
    

    print('M:ok')
