#!/usr/bin/env python3
''' problem:  Aer simu of majority voting for feed-forward
Fixed in :  qiskit-aer   0.13.2

'''
from pprint import pprint
from qiskit.tools.visualization import circuit_drawer
from qiskit import   Aer, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.classical import expr, types
from qiskit.tools.visualization import circuit_drawer


#...!...!....................
def create_ghz_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):  qc.cx(0, i)
    qc.measure_all()
    return qc

#...!...!....................
def create_FeedF_circuit(n=4,doHack= False):
    #this hack will be not needed  after this bug is fixed: https://github.com/Qiskit/qiskit-aer/issues/2020
    
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n)
    qc = QuantumCircuit(qr,cr)
    qc.h(0)
    for i in range(1, n-1): qc.cx(0, i)
    qc.barrier()
    qc.measure(range(3), range(3))
    qc.barrier()
    # Compute the majority using the defined expression
    #majority = (b0 and b1) or (b0 and b2) or (b1 and b2)
    if doHack:
        # for older Aer versions
        b01 = expr.bit_and(expr.cast(cr[0], types.Uint(1)), expr.cast(cr[1], types.Uint(1))) 
        b02 = expr.bit_and(expr.cast(cr[0], types.Uint(1)), expr.cast(cr[1], types.Uint(1)))  
        b12 = expr.bit_and(expr.cast(cr[1], types.Uint(1)), expr.cast(cr[2], types.Uint(1)))  
        bxx= expr.bit_or(b01, b02)
        majority = expr.cast(expr.bit_or(bxx, b12), types.Bool())
    else:
        b01=expr.bit_and(cr[0],cr[1]) # classical bits
        b02=expr.bit_and(cr[0],cr[2]) 
        b12=expr.bit_and(cr[1],cr[2]) 
        bxx= expr.bit_or(b01, b02)
        majority= expr.bit_or(bxx,b12)
    
    with qc.if_test(majority): qc.x(3) 

    qc.measure(3, 3)
    return qc
#=================================
if __name__ == "__main__":
     
    #qc=create_ghz_circuit(6)
    qc=create_FeedF_circuit()
    print(qc.draw(output='text',idle_wires=True,cregbundle=False))
     
    backend = Aer.get_backend('aer_simulator')

    job = backend.run(qc,shots=1000, dynamic=  True) 
    result=job.result()

    print('counts:',result.get_counts(0))
    #print(qc)
    #print(circuit_drawer(qc, output='text',cregbundle=False))
    
    print(qc.draw(output='text',idle_wires=True,cregbundle=False))

    print('M:ok')

