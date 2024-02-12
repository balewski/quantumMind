#!/usr/bin/env python3
''' problem:  write & read Qasm3

Should I use QPY for write/read binary the full circuit in the system agnostic way?
https://docs.quantum.ibm.com/api/qiskit/qpy

Or switch to a newer Qasm3: 'OQ3'
https://github.com/Qiskit/qiskit-qasm3-import



'''
from pprint import pprint
from qiskit import   QuantumCircuit, ClassicalRegister, QuantumRegister,transpile
from qiskit.circuit.classical import expr

from qiskit_ibm_runtime import QiskitRuntimeService
import qiskit.qasm3

#...!...!....................
def create_ghz_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):  qc.cx(0, i)
    qc.measure_all()
    return qc

#...!...!....................
def create_FeedF_circuit(n=4):
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n+1)
    qc = QuantumCircuit(qr,cr)
    qc.h(0)
    for i in range(1, n-1): qc.cx(0, i)
    qc.barrier()
    qc.measure(range(3), range(3))
    qc.barrier()
    
    # Compute the majority using the defined expression
    #majority = (b0 and b1) or (b0 and b2) or (b1 and b2)
    b01=expr.bit_and(cr[0],cr[1]) # classical bits
    b02=expr.bit_and(cr[0],cr[2]) # classical bits
    b12=expr.bit_and(cr[1],cr[2]) # classical bits
    bxx= expr.bit_or(b01, b02)
    majority= expr.bit_or(bxx,b12)
    with qc.if_test(majority): qc.x(3)

    qc.measure(3, 3)
    return qc
#=================================
if __name__ == "__main__":

    #qc=create_ghz_circuit(4)  # WORKS, but forgets the mapping
    qc=create_FeedF_circuit()  # CRASHES
    print(qc)

    #backName = "ibm_cairo"  #CX
    backName = "ibm_torino"  #CZ
    backName = "ibm_kyoto"   #ECR

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
    qasmInst=qiskit.qasm3.dumps(qcT)
    qasmF='out/myCirc_%s.qasm'%backend.name
    with open(qasmF, 'w') as file:
        file.write(qasmInst)        
    print('M: save one  transpiled circ:',qasmF)

    
    with open(qasmF, 'r') as file:
        qasm2 = file.read()
    print('\nM: dump read QASM:')
    print(qasm2)

    print('\nM: parse Qasm3')
    qc2=qiskit.qasm3.loads(qasm2)
    print('M: print mported qc2')
    print(qc2.draw(output='text',idle_wires=False,cregbundle=False)) 
    

    print('M:ok')
