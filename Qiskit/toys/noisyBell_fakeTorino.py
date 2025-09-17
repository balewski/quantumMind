#!/usr/bin/env python3

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino

def make_ghz(nq=3):
    circuit = QuantumCircuit(nq)
    circuit.h(0)
    for i in range(1,nq):
        circuit.cx(0, i)
    circuit.measure_all()  # Add measurements first
        
    return circuit

if __name__ == "__main__":

    qc = make_ghz()
    print(qc)
    
        
    nshot=2000    
    #------------------------------------------------
    # A: Simulate the circuit using fake Torino backend
    backend = FakeTorino()
    qcT = transpile(qc, backend)
    
    # Run the transpiled circuit using the chosen backend
    job = backend.run(qcT, shots=nshot)    
    counts = job.result().get_counts()
    print(backend.name,'Counts:',counts)

    #------------------------------------------------
    print('\n Simulate the circuit using an ideal Aer simulation')
    backend = AerSimulator()
    cx_u3_basis = ['cx', 'u3']
    qcT = transpile(qc, backend,basis_gates=cx_u3_basis)
    print(qcT)
    job = backend.run(qcT, shots=nshot)    
    counts = job.result().get_counts()
    print(backend.name,'Counts:',counts)
    

    
   
