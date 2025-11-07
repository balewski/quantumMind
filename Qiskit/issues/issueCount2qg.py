#!/usr/bin/env python3
# proper counting of 2q gates

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister 


def create_cnot_teleport_circuit(inp_ctrl, inp_targ, n_stack=1):
    """
    Create stacked CNOT teleportation circuit with computational basis input
    inp_ctrl, inp_targ: 0 or 1 for input computational basis state
    n_stack: number of CNOT teleportations to stack (default=1)
    """
    # Calculate total qubits needed: 2 (ctrl, trg) + 2*n_stack (ancillas)
    n_qubits = 2 + 2 * n_stack
    
    # Create control and target registers
    qctr = QuantumRegister(1, 'qctr')
    qtrg = QuantumRegister(1, 'qtrg')
    
    # Create ancilla registers for each teleportation
    anc_regs = [QuantumRegister(2, f'anc{i}') for i in range(n_stack)]
    
    # Create classical registers for teleportation measurements
    c_regs = [ClassicalRegister(2, f'ab{i}') for i in range(n_stack)]
    
    # Create classical register for final measurements
    freg = ClassicalRegister(2, 'ct')
    
    # Build circuit with all registers
    qc = QuantumCircuit(qctr, qtrg, *anc_regs, *c_regs, freg)
    
    # Prepare input states in computational basis
    if inp_ctrl:
        qc.x(qctr[0])  # control qubit = |1>
    if inp_targ:
        qc.x(qtrg[0])  # target qubit = |1>
    qc.barrier()
    
    # Stack n_stack teleported CNOTs
    for k in range(n_stack):
        anc = anc_regs[k]
        creg = c_regs[k]
        
        # Create entangled pair between processors
        qc.h(anc[0])
        qc.cx(anc[0], anc[1])
        qc.barrier()
        
        # Teleport CNOT gate
        qc.cx(qctr[0], anc[0])
        qc.cx(anc[1], qtrg[0])
        
        qc.h(anc[1])
        qc.measure(anc[0], creg[0])
        qc.measure(anc[1], creg[1])
        
        with qc.if_test((creg[0], 1)):
            qc.x(qtrg[0])  # X correction
        
        with qc.if_test((creg[1], 1)):
            qc.z(qctr[0])  # Z correction
        qc.barrier()
    
    # Measure the qubits in computational basis
    qc.measure(qctr[0], freg[0]) 
    qc.measure(qtrg[0], freg[1]) 
    
    return qc

           
   
if __name__ == "__main__":
    n_stack = 2
    print(f'Stacking {n_stack} CNOT teleportation(s)')
    # Create circuits for all  input states
    qc = create_cnot_teleport_circuit(0,1, n_stack)
    print(qc)

    #bad len2q = qc.depth(filter_function=lambda x: x.operation.num_qubits == 2)
    len2q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
    print(f'  len2q={len2q}, {qc.count_ops()}')
