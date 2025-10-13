#!/usr/bin/env python3
# symmetric circuit for CNOT teleportation, based on
# https://arxiv.org/pdf/1801.05283

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator

def create_cnot_teleport_serial(inpCT):
    """
    Create a qc for CNOT gate teleportation.
    Serial version
    The CNOT acts on qubits in different processors.
    """
    
    # Define quantum and classical registers
    # Processor A: control qubit (q0) and ancilla (q1)
    # Processor B: target qubit (q3) and ancilla (q2)
    qreg = QuantumRegister(4, 'q')
    creg = ClassicalRegister(2, 'ab')
    freg = ClassicalRegister(2, 'ct')
    qc = QuantumCircuit(qreg, creg,freg)    
    
    # Step 1: Prepare initial states (for testing)
    # Put control qubit in |1⟩ state and target in |0⟩

    if inpCT[0]: qc.x(qreg[0])  # Control = |1⟩
    if inpCT[1]: qc.x(qreg[3])  # Traget = |1⟩
    qc.barrier()
    
    # Step 2: Create entangled pair between processors
    qc.h(qreg[1])
    qc.cx(qreg[1], qreg[2])
    qc.barrier()

    # Step 3: Teleport CNOT gate
    qc.cx(qreg[0], qreg[1])
    qc.cx(qreg[2], qreg[3])
    
    qc.h(qreg[2])
    qc.measure(qreg[1], creg[0])
    qc.measure(qreg[2], creg[1])
    
    with qc.if_test((creg[0], 1)):
        qc.x(qreg[3])  # X correction
    
    with qc.if_test((creg[1], 1)):
        qc.z(qreg[0])  # Z correction
    qc.barrier()
    
    # now measure final qubits 0 & 3
    qc.measure(qreg[0], freg[1])
    qc.measure(qreg[3], freg[0])        
    return qc


# Create and visualize the qc

inpCT=(1,0)  # input (control, target)
qc = create_cnot_teleport_serial(inpCT)
print(qc.draw(output='text'))

# Simulate the qc
simulator = AerSimulator()
compiled_qc = transpile(qc, simulator)
job = simulator.run(compiled_qc, shots=1024)
result = job.result()
counts = result.get_counts(qc)

print('inp c,t:',inpCT)
print("Measurement results:\nct ab")
for outcome, count in sorted(counts.items()):
    print(f"{outcome}: {count}")

