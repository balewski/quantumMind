#!/usr/bin/env python3
# https://github.com/CQCL/pytket/tree/main/examples

print('M:start...')

from pytket.circuit import OpType
from pytket.extensions.qiskit import tk_to_qiskit
#...!...!....................
def print_qc(qc,text='myCirc'):
    cxDepth=qc.depth_by_type(OpType.CX)
    print('\n---- %s ---- CX depth=%d'%(text,cxDepth))
    print(tk_to_qiskit(qc))
    
# ------ Pyket  circuit -----
from pytket import Circuit
qc = Circuit(3,3)
qc.X(0); qc.X(1)
qc.add_gate(OpType.CnX, [0,1,2])
qc.measure_all()
print_qc(qc,'Master')

if 0:
    from pytket.extensions.qiskit import AerBackend
    tk_backend = AerBackend()

if 1:  # ------ run noisy IBMQ simulator
    from pytket.extensions.qiskit import IBMQEmulatorBackend
    tk_backend = IBMQEmulatorBackend(  backend_name='ibmq_jakarta',  instance='', )

shots=1000 ; optL=0 

# connect to the backend
qcT = tk_backend.get_compiled_circuit(qc, optimisation_level=optL) # is optL used?
print_qc(qcT,'Compiled optL=%d'%optL)

print('simu  shots=%d'%(shots),tk_backend) 
handle = tk_backend.process_circuit(qcT, n_shots=shots)
counts = tk_backend.get_result(handle).get_counts()  
print('M:counts',counts)

