#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# expands clbit register in different ways

# Function to print the circuit in text format
def print_circuit_text(circuit):
    for command in circuit:
        print(command)

from pytket.extensions.qiskit import tk_to_qiskit
from pytket import Circuit, Bit
nq=5
c = Circuit(nq,1)
for i in range(nq-1):  c.CX(i,i+1)
c.add_barrier([j for j in range(nq)])
c.Measure(0,0)

print('\nM----- A')
print(tk_to_qiskit(c))


c.add_bit(Bit("c", 1))
c.add_bit(Bit("c", 2))
c.Measure(c.qubits[1],c.bits[1])
c.Measure(c.qubits[2],c.bits[2])

print('\nM----- B')
print(tk_to_qiskit(c))


c.add_c_register(name="d",size=2)
c.Measure(c.qubits[3],c.bits[3])
c.Measure(c.qubits[4],c.bits[4])
print('\nM----- C')
print(tk_to_qiskit(c))


