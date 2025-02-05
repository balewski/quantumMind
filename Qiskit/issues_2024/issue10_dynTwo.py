#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# issue: qc.draw() fails for 3-qubit circuit with 3 bit-conditioned gates
# ticket:

import qiskit as qk
import qiskit.qasm3
import sys


#...!...!....................
def circMcbXM(nq):
    crPre = qk.ClassicalRegister(nq, name="pre")
    crPost = qk.ClassicalRegister(nq, name="post")
    qr = qk.QuantumRegister(nq, name="q")
    qc= qk.QuantumCircuit(qr, crPre,crPost,name='McbXM')
    for i in range(nq):
        qc.h(i)
        qc.measure(i,crPre[i])
        with qc.if_test((crPre[i], 0)):
            qc.x(i)  # conditionla state flip
        qc.measure(i, crPost[i])
    return qc

# -------Create a Quantum Circuit 
nq=3
qc=circMcbXM(nq)
print('created circ for nq=',nq)

print('M: dump QASM3 ideal circ:\n')
qiskit.qasm3.dump(qc,sys.stdout)
print('\n  --- end ---\n')
print(qc.draw(output="text", idle_wires=False))
