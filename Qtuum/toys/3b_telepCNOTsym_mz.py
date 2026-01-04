#!/usr/bin/env python3
# symmetric circuit for CNOT teleportation, based on
# https://arxiv.org/pdf/1801.05283

import os, secrets
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
import qnexus as qnx
from pytket.circuit import BasisOrder
from time import time
from pprint import pprint

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


#=================================
#  M A I N 
#=================================
if __name__ == "__main__":
    inpCT=(1,0)  # input (control, target)
    qc_qiskit = create_cnot_teleport_serial(inpCT)
    print("Qiskit circuit:")
    print(qc_qiskit.draw(output='text'))

    # --- Convert to TKET ---
    print("\nConverting to TKET...")
    qc_tket = qiskit_to_tk(qc_qiskit)
    print(qc_tket)
    print("\n--- Gate Sequence in TKet---")
    for command in qc_tket.get_commands():
        print(command)
    print(tk_to_qiskit(qc_tket))

    # --- QNexus Setup ---
    myTag = '_' + secrets.token_hex(3)
    shots = 100
    #devName = "H1-Emulator"   # noisy
    devName = "H1-1LE"        # noiseless
    myAccount = 'CSC641'
    project = qnx.projects.get_or_create(name="qcrank-jan-04")
    qnx.context.set_active_project(project)

    print(f"\nuploading circuit to Nexus (tag: {myTag}) ...")
    t0 = time()
    ref = qnx.circuits.upload(circuit=qc_tket, name='telepCNOT'+myTag)
    t1 = time()
    print('elaT=%.1f sec, uploaded, compiling ...'%(t1-t0))

    devConf = qnx.QuantinuumConfig(device_name=devName, user_group=myAccount) 
    
    t0 = time()
    #...  compile 
    refC_list = qnx.compile(programs=[ref], name='comp'+myTag,
                            optimisation_level=2, backend_config=devConf,
                            project=project)
    refC = refC_list[0]
    t1 = time()
    print('elaT=%.1f sec, compiled, executing ...'%(t1-t0))

    #... get cost
    cost = qnx.circuits.cost(circuit_ref=refC, n_shots=shots,
                             backend_config=devConf, syntax_checker="H1-1SC")
    print('\nshots=%d cost=%.1f:'%(shots, cost))

    #.... execution     
    t0 = time()
    ref_exec = qnx.start_execute_job(programs=[refC], n_shots=[shots],
                                     backend_config=devConf, name="exec"+myTag)    
    t1 = time()
    print('job submit elaT=%.1f, waiting for results ...'%(t1-t0))
    
    qnx.jobs.wait_for(ref_exec)
    results = qnx.jobs.results(ref_exec)
    t2 = time()
    print('execution finished, total elaT=%.1f\n'%(t2-t1))
    
    result = results[0].download_result()
    
    # DLO gives bits in order defined in the circuit: c[n-1], ..., c[0]
    tket_counts = result.get_counts(basis=BasisOrder.dlo)
    
    print('inp c,t:',inpCT)
    print("\nMeasurement results (TKET DLO order):")
    pprint(tket_counts)
    
    print('\ndone devName=', devName)
    print('status:', qnx.jobs.status(ref_exec))

