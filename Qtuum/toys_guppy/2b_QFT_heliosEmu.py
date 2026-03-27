#!/usr/bin/env python3
"""
QFT benchmark with noisy local simulation using Guppy.
This script compares an ideal simulation with a noisy statevector simulation.
The circuit prepares state |k> in the Fourier basis and then applies IQFT.
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
from collections import Counter
from pprint import pprint

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import h, rz, crz, cx, measure, measure_array, qubit, pi, array
from guppylang.std.mem import mem_swap

import os
import secrets
from time import time
import qnexus as qnx

from toolbox.Util_Guppy import guppy_to_qiskit

# ---- Guppy circuit definitions ----

INP_INT =25
NQ_VAL = 24  # allowed 6,10,14,18,22,24,26
nq = guppy.nat_var("nq")

@guppy
def iqft_n(qs: array[qubit, nq]) -> None:
    """Standard Inverse QFT on nq qubits matching Qiskit's QFTGate.inverse()"""
    # 1. Reverse qubit order with physical SWAP gates
    for i in range(nq // 2):
        cx(qs[i], qs[nq - 1 - i])
        cx(qs[nq - 1 - i], qs[i])
        cx(qs[i], qs[nq - 1 - i])
    
    # 2. Iterate through qubits and apply rotations and H gates
    for i in range(nq):
        # Apply controlled rotations from previous qubits
        # We must build a true CP(phi) gate.
        # CP(c, t, phi) is equivalent to Rz(c, phi/2) + CRz(c, t, phi) up to global phase.
        for j in range(i):
            phi = -pi / (2 ** (i - j))
            rz(qs[j], phi / 2)
            crz(qs[j], qs[i], phi)
        
        # Apply H gate
        h(qs[i])

@guppy
def qft_prep_n(qs: array[qubit, nq], inpInt: int) -> None:
    """Prepares Fourier state corresponding to computational state |inpInt>."""
    # 1. Apply H on all
    for i in range(nq):
        h(qs[i])
    
    # 2. Apply relative phases based on inpInt
    for j in range(nq):
        rz(qs[j], (2.0 * pi * inpInt) / (2 ** (nq - j)))

def get_main_bench(nq: int):
    """Returns a Guppy program configured for the specific nq using if/else statements."""
    if nq == 6:
        @guppy
        def main_qft_bench() -> None:
            inpInt = comptime(INP_INT)
            qs = array(qubit(), qubit(), qubit(), qubit(), qubit(), qubit())
            qft_prep_n(qs, inpInt)
            iqft_n(qs)
            bits = measure_array(qs)
            result("b0", bits[0])
            result("b1", bits[1])
            result("b2", bits[2])
            result("b3", bits[3])
            result("b4", bits[4])
            result("b5", bits[5])
        return main_qft_bench

    elif nq == 10:
        @guppy
        def main_qft_bench() -> None:
            inpInt = comptime(INP_INT)
            qs = array(qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit())
            qft_prep_n(qs, inpInt)
            iqft_n(qs)
            bits = measure_array(qs)
            result("b0", bits[0])
            result("b1", bits[1])
            result("b2", bits[2])
            result("b3", bits[3])
            result("b4", bits[4])
            result("b5", bits[5])
            result("b6", bits[6])
            result("b7", bits[7])
            result("b8", bits[8])
            result("b9", bits[9])
        return main_qft_bench

    elif nq == 14:
        @guppy
        def main_qft_bench() -> None:
            inpInt = comptime(INP_INT)
            qs = array(qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit())
            qft_prep_n(qs, inpInt)
            iqft_n(qs)
            bits = measure_array(qs)
            result("b0", bits[0])
            result("b1", bits[1])
            result("b2", bits[2])
            result("b3", bits[3])
            result("b4", bits[4])
            result("b5", bits[5])
            result("b6", bits[6])
            result("b7", bits[7])
            result("b8", bits[8])
            result("b9", bits[9])
            result("b10", bits[10])
            result("b11", bits[11])
            result("b12", bits[12])
            result("b13", bits[13])
        return main_qft_bench

    elif nq == 18:
        @guppy
        def main_qft_bench() -> None:
            inpInt = comptime(INP_INT)
            qs = array(qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit())
            qft_prep_n(qs, inpInt)
            iqft_n(qs)
            bits = measure_array(qs)
            result("b0", bits[0])
            result("b1", bits[1])
            result("b2", bits[2])
            result("b3", bits[3])
            result("b4", bits[4])
            result("b5", bits[5])
            result("b6", bits[6])
            result("b7", bits[7])
            result("b8", bits[8])
            result("b9", bits[9])
            result("b10", bits[10])
            result("b11", bits[11])
            result("b12", bits[12])
            result("b13", bits[13])
            result("b14", bits[14])
            result("b15", bits[15])
            result("b16", bits[16])
            result("b17", bits[17])
        return main_qft_bench

    elif nq == 22:
        @guppy
        def main_qft_bench() -> None:
            inpInt = comptime(INP_INT)
            qs = array(qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit())
            qft_prep_n(qs, inpInt)
            iqft_n(qs)
            bits = measure_array(qs)
            result("b0", bits[0])
            result("b1", bits[1])
            result("b2", bits[2])
            result("b3", bits[3])
            result("b4", bits[4])
            result("b5", bits[5])
            result("b6", bits[6])
            result("b7", bits[7])
            result("b8", bits[8])
            result("b9", bits[9])
            result("b10", bits[10])
            result("b11", bits[11])
            result("b12", bits[12])
            result("b13", bits[13])
            result("b14", bits[14])
            result("b15", bits[15])
            result("b16", bits[16])
            result("b17", bits[17])
            result("b18", bits[18])
            result("b19", bits[19])
            result("b20", bits[20])
            result("b21", bits[21])
        return main_qft_bench

    elif nq == 24:
        @guppy
        def main_qft_bench() -> None:
            inpInt = comptime(INP_INT)
            qs = array(qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit())
            qft_prep_n(qs, inpInt)
            iqft_n(qs)
            bits = measure_array(qs)
            result("b0", bits[0])
            result("b1", bits[1])
            result("b2", bits[2])
            result("b3", bits[3])
            result("b4", bits[4])
            result("b5", bits[5])
            result("b6", bits[6])
            result("b7", bits[7])
            result("b8", bits[8])
            result("b9", bits[9])
            result("b10", bits[10])
            result("b11", bits[11])
            result("b12", bits[12])
            result("b13", bits[13])
            result("b14", bits[14])
            result("b15", bits[15])
            result("b16", bits[16])
            result("b17", bits[17])
            result("b18", bits[18])
            result("b19", bits[19])
            result("b20", bits[20])
            result("b21", bits[21])
            result("b22", bits[22])
            result("b23", bits[23])
        return main_qft_bench

    elif nq == 26:
        @guppy
        def main_qft_bench() -> None:
            inpInt = comptime(INP_INT)
            qs = array(qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit(), qubit())
            qft_prep_n(qs, inpInt)
            iqft_n(qs)
            bits = measure_array(qs)
            result("b0", bits[0])
            result("b1", bits[1])
            result("b2", bits[2])
            result("b3", bits[3])
            result("b4", bits[4])
            result("b5", bits[5])
            result("b6", bits[6])
            result("b7", bits[7])
            result("b8", bits[8])
            result("b9", bits[9])
            result("b10", bits[10])
            result("b11", bits[11])
            result("b12", bits[12])
            result("b13", bits[13])
            result("b14", bits[14])
            result("b15", bits[15])
            result("b16", bits[16])
            result("b17", bits[17])
            result("b18", bits[18])
            result("b19", bits[19])
            result("b20", bits[20])
            result("b21", bits[21])
            result("b22", bits[22])
            result("b23", bits[23])
            result("b24", bits[24])
            result("b25", bits[25])
        return main_qft_bench

    else:
        raise ValueError(f"nq={nq} is not supported. Must be in [6, 10, 14, 18, 22, 24, 26].")


# ---- Helpers for simulation ----
 
def submit_job(hugr_pkg, nq: int, shots: int = 10,
               dev_name: str = "Helios-1E",
               account: str = "CHM170",
               project_name: str = "feb-guppy",
               max_cost: int = 500):
    tag = "_" + secrets.token_hex(3)

    project = qnx.projects.get_or_create(name=project_name)
    qnx.context.set_active_project(project)

    # Upload HUGR
    t0 = time()
    ref = qnx.hugr.upload(
        hugr_package=hugr_pkg,
        name=f"jan-qft-{tag}",
        description="QFT benchmark on Helios from Guppy",
        project=project,
    )
    print("upload elaT=%.1f sec" % (time() - t0))

    dev_conf = qnx.QuantinuumConfig(
        device_name=dev_name,
        user_group=account,
        max_cost=max_cost,
        compiler_options={"max-qubits": nq},
    )
    print("devConf:", dev_conf)

    # Submit execution job
    t0 = time()
    ref_exec = qnx.start_execute_job(
        programs=[ref], n_shots=[shots],
        backend_config=dev_conf, name="exec" + tag,
    )
    print("job submit elaT=%.1f sec" % (time() - t0))
    #1print("job ref:", ref_exec)
    return ref_exec, tag

def retrieve_job(ref_exec, nq):
    t0 = time()
    print("waiting for job to complete ...")
    qnx.jobs.wait_for(ref_exec)
    wait_time = time() - t0

    status = qnx.jobs.status(ref_exec)
    results = qnx.jobs.results(ref_exec)
    result_data = results[0].download_result()

    exec_sec = 0.0
    if hasattr(status, 'completed_time') and hasattr(status, 'running_time') and status.completed_time and status.running_time:
        exec_sec = (status.completed_time - status.running_time).total_seconds()

    meta_dict = {
        'wait_sec': round(wait_time, 1),
        'n_shots': len(result_data.results),
        'exec_sec': round(exec_sec, 1),
        'num_qubit': nq,
        'cost_hqc': status.cost,
        'job_id': ref_exec.id,
        
    }

    print("\n--- Job metadata ---")
    print(meta_dict)
    return result_data

def run_ideal(guppy_prog, nq, shots, seed=42):
    # QFT requires statevector_sim because of arbitrary rotations.
    emulator = guppy_prog.emulator(n_qubits=nq).statevector_sim().with_seed(seed)
    return emulator.with_shots(shots).run()

def postprocess_shots(sim_result, nq, correct_int=None, label=""):
    if correct_int is None:
        correct_int = INP_INT
        
    shots = sim_result.results
    total_shots = len(shots)
    print(f"\n--- {label}: {total_shots} shots ---")
    
    counts = Counter()
    for shot in shots:
        entries = {name: val for name, val in shot.entries}
        # Construct bitstring. Qiskit style: MSB on the left -> b5 b4 b3 b2 b1 b0
        bitstr = "".join(str(entries[f"b{i}"]) for i in reversed(range(nq)))
        counts[bitstr] += 1
    
    success_shots = 0
    # Sort and print
    npr=0
    for bitstr, cnt in counts.most_common():
        val = int(bitstr, 2)
        if val == correct_int:  success_shots += cnt
        npr+=1
        if npr>5: break
        print(f"  {bitstr} (dec={val:2d}): {cnt}")

    if total_shots > 0:
        incorrect_shots = total_shots - success_shots
        prob = success_shots / total_shots
        err_prob = incorrect_shots / total_shots
        
        incorrect_states_seen = sum(1 for bitstr in counts if int(bitstr, 2) != correct_int)
        total_states_space = 2**nq
        err_err = np.sqrt(prob * (1 - prob) / total_shots)
        
        print(f"\nIncorrect states: {incorrect_states_seen} of {total_states_space}, total probability: {err_prob:.4f} ({incorrect_shots}/{total_shots})")
        print(f"QFT nq:{nq}   prob: {prob:.4f} +/- {err_err:.4f},   Success: {success_shots}\n")

# ---- Main Execution ----

def main():


    # 1. Check Guppy Program
    main_qft_bench = get_main_bench(NQ_VAL)
    print("\nChecking Guppy program...")
    print("  iqft_n check:", iqft_n.check())
    print("  qft_prep_n check:", qft_prep_n.check())
    print("  main_qft_bench check:", main_qft_bench.check())


    if NQ_VAL <5:
        circQi=guppy_to_qiskit(main_qft_bench,nq=NQ_VAL)
        print(circQi)
   
    # 3. Execution
    num_shots = 200
    if NQ_VAL<23:        
        print(f"\nRunning ideal simu locally (inpInt={INP_INT}, nq={NQ_VAL}, shots={num_shots}) ...")
    
        # Ideal
        t0 = time()
        ideal_result = run_ideal(main_qft_bench, nq=NQ_VAL, shots=num_shots)
        elaT = time() - t0
        print(f"ideal simu elaT={elaT:.1f} sec")
        postprocess_shots(ideal_result, NQ_VAL, label="Ideal")

    
    # Helios Emulator
    
    print("\nCompiling to HUGR IR for Helios...")
    hugr_pkg = main_qft_bench.compile()
    
    print(f"Submitting {num_shots} shots to Helios-1E...")
    ref_exec, tag = submit_job(hugr_pkg, nq=NQ_VAL, shots=num_shots, dev_name="Helios-1E")
    
    result_data = retrieve_job(ref_exec, nq=NQ_VAL)
    if result_data:
        postprocess_shots(result_data, NQ_VAL, label="HeliosEmu")

if __name__ == "__main__":
    main()
    os._exit(0)  # Bypass native teardown to prevent segfaults from qnexus/pytket on exit
