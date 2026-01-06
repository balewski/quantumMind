#!/usr/bin/env python3
# symmetric circuit for CNOT teleportation, based on
# https://arxiv.org/pdf/1801.05283

import os, secrets
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from pytket.extensions.qiskit import qiskit_to_tk  # tk_to_qiskit is not used here
import qnexus as qnx
from pytket.circuit import BasisOrder
from time import time
from pprint import pprint

def create_cnot_teleport(inpCT):
    """
    Create a qc for CNOT gate teleportation.
    The CNOT acts on qubits in different processors.
    """
    
    # Define quantum and classical registers
    # Processor A: control qubit (q0) and ancilla (q1)
    # Processor B: target qubit (q3) and ancilla (q2)
    qreg = QuantumRegister(4, 'q')
    mreg = ClassicalRegister(2, 'ab')  # mid-circuit measurement
    freg = ClassicalRegister(2, 'ct')  # final register
    qc = QuantumCircuit(qreg,freg,mreg)    
    
    # Step 1: Prepare initial states (for testing)
    if inpCT[0]: qc.x(qreg[0])  # Control = |1⟩
    if inpCT[1]: qc.x(qreg[3])  # Target = |1⟩
    qc.barrier()
    
    # Step 2: Create entangled pair between processors
    qc.h(qreg[1])
    qc.cx(qreg[1], qreg[2])
    qc.barrier()

    # Step 3: Teleport CNOT gate
    qc.cx(qreg[0], qreg[1])
    qc.cx(qreg[2], qreg[3])
    
    qc.h(qreg[2])
    qc.measure(qreg[1], mreg[0])
    qc.measure(qreg[2], mreg[1])
    
    with qc.if_test((mreg[0], 1)):
        qc.x(qreg[3])  # X correction
    
    with qc.if_test((mreg[1], 1)):
        qc.z(qreg[0])  # Z correction
    qc.barrier()
    
    # now measure final qubits 0 & 3
    qc.measure(qreg[0], freg[1])
    qc.measure(qreg[3], freg[0])      
    return qc
def get_creg_list(qc_tket):
    """
    Extracts the alphabetical order of registers and their sizes from the TKet circuit.
    Necessary because TKet sorts registers during conversion.
    """
    from itertools import groupby
    return [[name, len(list(group))] for name, group in groupby(qc_tket.bits, key=lambda x: x.reg_name)]

def split_tket_counts_by_creg(creg_list, tket_counts):
    """
    Splits the combined TKET counts into counts per classical register.
    Takes as input a list of (register_name, size) tuples.
    Returns a double dictionary: {reg_name: {bitstring: count}}
    """
    # Initialize the results container
    reg_counts = {name: {} for name, size in creg_list}
    
    for bit_tuple, count in tket_counts.items():
        # TKet DLO basis: index 0 is MSB (last bit in definition list).
        # Reverse bit_tuple to align indices with alphabetical definition order.
        rev_tuple = bit_tuple[::-1]
        
        ptr = 0
        for name, size in creg_list:
            # Slice the reversed tuple for this register
            chunk = rev_tuple[ptr : ptr + size]
            # Convert bit tuple to string. Qiskit strings are bit[n-1]...bit[0] (Big-Endian)
            # chunk is (bit[0], ... bit[n-1]), so we reverse it
            bitstr = "".join(str(b) for b in reversed(chunk))
            
            reg_counts[name][bitstr] = reg_counts[name].get(bitstr, 0) + count
            ptr += size
            
    return reg_counts

#=================================
#  M A I N 
#=================================
if __name__ == "__main__":
    inpCT = (1,0)   # input (control, target)
    qc_qiskit = create_cnot_teleport(inpCT)
    print('inp c,t:', inpCT) 
    print("Qiskit circuit:")
    print(qc_qiskit.draw(output='text'))

    # --- Convert to TKET ---
    print("\nConverting to TKET...")
    qc_tket = qiskit_to_tk(qc_qiskit)
    print(qc_tket)

    # Alphabetical order of registers in TKet
    creg_list = get_creg_list(qc_tket)
    print('creg_list:', creg_list)

    print("\n--- Gate Sequence in TKet---")
    for command in qc_tket.get_commands():
        print(command)

    # --- QNexus Setup ---
    myTag = '_' + secrets.token_hex(3)
    shots = 1000
    devName = "H2-1E"
    myAccount = 'CSC641'
    project = qnx.projects.get_or_create(name="qcrank-feb")
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
    
    # DLO gives bits in order defined in the circuit: last_register, ..., first_register
    tket_counts = result.get_counts(basis=BasisOrder.dlo)
    
    print('inp c,t:', inpCT)
    print("\nCombined Measurement results (TKET DLO order):")
    pprint(tket_counts)
    
    # --- Split results by Classical Register --- 
    print("\nSplit results by Classical Register:")
    split_counts = split_tket_counts_by_creg(creg_list, tket_counts)
    pprint(split_counts)
    
    print('\ndone devName=', devName)
    print('status:', qnx.jobs.status(ref_exec))
