#!/usr/bin/env python3
#from Aziz: Logical expression can be used on both classiclal registers and classical bits

# Expresions:  https://docs.quantum.ibm.com/api/qiskit/0.44/circuit_classical

import numpy as np
from qiskit.circuit.classical import expr  # feed-forward
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.visualization import circuit_drawer
from time import time, sleep
from qiskit.providers.jobstatus import JobStatus


qr = QuantumRegister(4)
cr1 = ClassicalRegister(2)
cr2 = ClassicalRegister(2)

qc = QuantumCircuit(qr,cr1, cr2)
qc.ry(np.pi/3,range(4))
qc.measure(range(4), range(4))

if 1:  # skip conditional logic
    cond1 = expr.bit_or(cr1, cr2)  # <== defines a logical condition on clreg

    with qc.if_test(expr.equal(cond1, 2)): # classical registers (aka many bits interpreted as int)
        qc.cx(0,1)
        qc.cx(2,3)

    with qc.if_test(expr.bit_and(cr1[0],cr2[1])): # classical bits
        qc.cx(3,2)
        qc.h(0)
    qc.measure(range(4), range(4))

print(circuit_drawer(qc.decompose(), output='text',cregbundle=False, idle_wires=False))
backendN="ibmq_qasm_simulator"
#backendN="ibm_cusco"

# Aziz: backend.run() can be invoked directly now from qiskit_ibm_runtime. No longer need t use IBMProvider().

if 1:
    print('M:IBMProvider()...')
    from qiskit_ibm_provider import IBMProvider
    provider = IBMProvider()    
    backend = provider.get_backend(backendN)
else:
    print('M: access QiskitRuntimeService()...')
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    backend = service.get_backend(backendN)
print('use backend version:',backend.version )

if 0:
    assert  backendN!="ibmq_qasm_simulator"
else:
    from qiskit import   Aer
    backend = Aer.get_backend('aer_simulator')

'''
a) b) The ibmq_qasm_simulator does not support dynamic circuits(just a few instructions) and there is no plan to fylly support dynamic circuits on this backend. Sorry for this inconvenience. Instead we recommend qiskit-aer for simulation.
''' 


# from Aziz, patch backend for feed-forward 'NEW' logic
#  with qc.if_test((qc.cregs[0], 1)): qc.h(2)  # NEW
#  an issue opened on github https://github.com/Qiskit/qiskit-ibm-runtime/issues/1253. 
if  "if_else" not in backend.target:
    from qiskit.circuit import IfElseOp
    backend.target.add_instruction(IfElseOp, name="if_else")

print('M: transpiling ...',backend)
qcT = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=42)
print('M: run ...')
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
