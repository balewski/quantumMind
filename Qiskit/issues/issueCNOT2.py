#!/usr/bin/env python3
# circuit implements CNOT telportation, follwoing fig 1c in https://arxiv.org/pdf/1801.05283

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister 
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeCusco, FakeBrisbane
import argparse
import time

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

def get_parser():
    parser = argparse.ArgumentParser(
        description='CNOT teleportation truth table',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-n','--nshot', type=int, default=1_000,
                        help='Number of shots for circuit execution')
    parser.add_argument('-b','--backendType', type=int, default=3, 
                        help='Backend type: 1=ideal (AerSimulator), 1=FakeTorino')

    return parser

def main(args):
    nshot = args.nshot
    
    # Select backend based on backendType argument
    if args.backendType == 0:
        backend = AerSimulator()
        print('Backend: AerSimulator (ideal)')
    elif args.backendType == 1:
        backend = FakeTorino()
        print('Backend: FakeTorino')
    elif args.backendType == 2:
        backend = FakeBrisbane()
    elif args.backendType == 3:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService() #channel="ibm_quantum_platform")
        backName='ibm_fez'
        print('\n real HW   %s backend ...'%backName)
        backend = service.backend(backName)

    n_stack = 2
    print(f'Stacking {n_stack} CNOT teleportation(s)')
    # Create circuits for all  input states
    circuits = []
    inp_targ=0
    for inp_ctrl in [0, 1]:
        qc = create_cnot_teleport_circuit(inp_ctrl, inp_targ, n_stack)
        circuits.append(qc)
        print(qc)
        #break

    print(f'Created {len(circuits)} circuits for truth table (stack={n_stack})')
    print(f'Transpiling ...  for {backend.name}')
 
    # Transpile all circuits
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    print('aa')
    qcT = pm.run(circuits,num_processes=1)     

    print(f'Transpiled circuits for {backend.name}')
    print(f'Number of qubits per circuit: {qcT[0].num_qubits}')
  

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
