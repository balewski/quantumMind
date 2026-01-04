#!/usr/bin/env python3

import numpy as np
import time
import argparse
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.circuit import MidCircuitMeasure
from qiskit.circuit import Measure
from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

if __name__ == "__main__":
    nshot = 100
 
    # Setup quantum and classical registers
    ctrl_q = QuantumRegister(1, "ctrl")   # control qubit
    trg_q = QuantumRegister(1, "trg")     # target qubit
    creg = ClassicalRegister(1, "creg")   # classical register to store mid-measure result

    qc = QuantumCircuit(ctrl_q, trg_q, creg)

    # Step 1: Prepare a superposition on control qubit
    qc.h(ctrl_q[0])  

    # Step 2: Explicit Mid-Circuit Measurement
    mid_instr = MidCircuitMeasure()
    qc.append(mid_instr, [ctrl_q[0]], [creg[0]])

    # Step 3: Conditional X correction (CX equivalent)
    # If creg[0] == 1, flip the target qubit
    with qc.if_test((creg, 1)):
        qc.x(trg_q[0])

    # Step 4: Final measurements for verification
    qc.measure_all()

    print("\nOriginal Circuit:")
    print(qc.draw(output="text"))

    if 1:
        backend = AerSimulator()
        print('Backend: AerSimulator (ideal)')
        #qcTL=qc
    else:
        backName = 'ibm_pittsburgh'
        #backName = 'ibm_kingston'
        
        service = QiskitRuntimeService()
        backL=service.backends(filters=lambda b: "measure_2" in b.supported_instructions)
        print('measure_2' ,backL)

        print('\n real HW   %s backend ...' % backName)
        backend = service.backend(backName)
    
    # --- Transpile --- 
    pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend)
    qcTL = pass_manager.run(qc)

    print("\nTranspiled Circuit:")
    print(qcTL.draw(output="text"))

    # --- Run with SamplerV2 ---
    sampler = Sampler(mode=backend)
    t0 = time.time()
    job = sampler.run([qcTL], shots=nshot)
    result = job.result()
    t1 = time.time()

    print(f"\nJob finished in {t1 - t0:.2f} seconds.")

    pub_result = result[0]
    counts = pub_result.data.c_final.get_counts()

    print("\nFinal measurement counts (from c_final):")
    print(counts)
