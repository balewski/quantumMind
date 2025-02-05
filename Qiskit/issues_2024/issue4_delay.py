#!/usr/bin/env python3
''' problem: adding delay inside corcuit results with 

'''
import qiskit as qk
import qiskit.qasm3

# -------Create a Quantum Circuit 
circ = qk.QuantumCircuit(3,3)
circ.h(0)
circ.cx(0, 1) 
circ.delay(400)
circ.cx(0, 2)
circ.barrier()
circ.measure(0,0)
print(circ)
print(qiskit.qasm3.dumps(circ))

print('M:ok')

