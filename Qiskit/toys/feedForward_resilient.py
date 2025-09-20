#!/usr/bin/env python3
"""
Feed-forward operations with majority voting for readout error compensation
"""

# use majority voting to compensate readout error for feed-forward
# Expresions:  https://docs.quantum.ibm.com/api/qiskit/0.44/circuit_classical

import numpy as np
from qiskit.circuit.classical import expr
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.tools.visualization import circuit_drawer
from qiskit_ibm_runtime import QiskitRuntimeService
from time import time, sleep
from qiskit.providers.jobstatus import JobStatus

def feedF_circuit(n=4):
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n)
    qc = QuantumCircuit(qr,cr)
    
    for i in range(0, n-1):
        qc.h(i)
    qc.barrier()
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.measure(2, 2)
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

#...!...!....................
def create_FeedF_circuit(n=4):
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n+1)
    qc = QuantumCircuit(qr,cr)
    qc.h(0)
    for i in range(1, n-1): qc.cx(0, i)
    qc.barrier()
    qc.measure(range(4), range(4))
    if 1:
        qc.measure(1, 4)
    qc.barrier()
    # Compute the majority using the defined expression
    #majority = (b0 and b1) or (b0 and b2) or (b1 and b2)
    b01=expr.bit_and(cr[0],cr[1]) # classical bits
    b02=expr.bit_and(cr[0],cr[2]) # classical bits
    b12=expr.bit_and(cr[1],cr[2]) # classical bits
    bxx= expr.bit_or(b01, b02)
    majority= expr.bit_or(bxx,b12)
    #majority=expr.bit_and(cr[0],cr[1])
    with qc.if_test(majority): qc.x(3) 

    qc.measure(3, 3)
    return qc

#...!...!....................
def create_some_circuit0():

    qr = QuantumRegister(4)
    cr1 = ClassicalRegister(2)
    cr2 = ClassicalRegister(2)

    qc = QuantumCircuit(qr,cr1, cr2)
    qc.ry(np.pi/3,range(4))
    qc.measure(range(4), range(4))
    cond1 = expr.bit_or(cr1, cr2)  # <== defines a logical condition on clreg

    with qc.if_test(expr.equal(cond1, 2)): # classical registers
        qc.cx(0,1)
        qc.cx(2,3)

    with qc.if_test(expr.bit_and(cr1[0],cr2[1])): # classical bits
        qc.cx(3,2)
        qc.h(0)
    qc.measure(range(4), range(4))
    return qc

#...!...!....................
def create_some_circuit():

    qr = QuantumRegister(4)
    cr1 = ClassicalRegister(2)
    cr2 = ClassicalRegister(2)

    qc = QuantumCircuit(qr,cr1, cr2)
    qc.ry(np.pi/3,range(4))
    qc.measure(range(4), range(4))
    qc.barrier()
     
    cond1 = expr.bit_or(cr1, cr2)  # <== defines a logical condition on clreg

    with qc.if_test(expr.equal(cond1, 2)): # classical registers
        qc.cx(0,1)
        qc.cx(2,3)

    with qc.if_test(expr.bit_and(cr1[0],cr2[1])): # classical bits
        qc.cx(3,2)
        qc.h(0)

    majority=expr.bit_and(cr1[0],cr1[1])
    with qc.if_test(majority): qc.x(3) 
    qc.measure(range(4), range(4))
    return qc

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":

    #qc=create_FeedF_circuit()
    #qc=create_some_circuit()
    qc=feedF_circuit()
    print(circuit_drawer(qc.decompose(), output='text',cregbundle=False, idle_wires=False))

    print('M: access QiskitRuntimeService()...')
    service = QiskitRuntimeService()
    backendN="ibmq_qasm_simulator"
    backendN="ibm_cusco"
    #backendN="ibm_kyoto"
    #backendN="ibm_hanoi"

    if 0:
        backend = service.get_backend(backendN)
    else:
        from qiskit import   Aer
        backend = Aer.get_backend('aer_simulator')
    
    print('use backend version:',backend.version )


    # from Aziz, patch backend for feed-forward 'NEW' logic
    #  with qc.if_test((qc.cregs[0], 1)): qc.h(2)  # NEW
    #  an issue opened on github https://github.com/Qiskit/qiskit-ibm-runtime/issues/1253. 
    if "if_else" not in backend.target:
        from qiskit.circuit import IfElseOp
        backend.target.add_instruction(IfElseOp, name="if_else")
    
    print('M: transpiling ...')
    qcT = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=42)
    job = backend.run(qcT,shots=1000, dynamic=True)
    
    i=0; T0=time()
    while True:
        jstat=job.status()
        elaT=time()-T0
        print('P:i=%d  status=%s, elaT=%.1f sec'%(i,jstat,elaT))
        if jstat in [ JobStatus.DONE, JobStatus.ERROR]: break
        i+=1; sleep(5)

    print('M: job done, status:',jstat,backend)

    result = job.result()
    counts=result.get_counts(0)
    print('M: counts:', counts)
