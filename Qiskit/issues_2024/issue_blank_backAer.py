#!/usr/bin/env python3
''' problem: ??

'''
from pprint import pprint
from qiskit.tools.visualization import circuit_drawer
from qiskit import   Aer, QuantumCircuit

#...!...!....................
def create_ghz_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):  qc.cx(0, i)
    qc.measure_all()
    return qc


#=================================
if __name__ == "__main__":
     
    qc=create_ghz_circuit(6)

    print(qc)

    backend = Aer.get_backend('aer_simulator')

    job = backend.run(qc,shots=1000, dynamic=True) 
    result=job.result()

    print('counts:',result.get_counts(0))

    print('M:ok')

