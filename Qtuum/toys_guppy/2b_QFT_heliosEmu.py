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

from guppylang import guppy
from guppylang.std.builtins import result, comptime
from guppylang.std.quantum import h, rz, crz, cx, measure_array, qubit, pi, array

import os
import secrets
from time import time
import qnexus as qnx

from toolbox.Util_Guppy import guppy_to_qiskit

# ---- Guppy circuit definitions ----

def get_main_bench(n_q: int, inp_int: int):
    """Returns a Guppy program specialized to the requested qubit count and input."""

    @guppy
    def iqft_n(qs: array[qubit, comptime(n_q)]) -> None:
        """Standard Inverse QFT on n_q qubits matching Qiskit's QFTGate.inverse()."""
        for i in range(comptime(n_q // 2)):
            cx(qs[i], qs[comptime(n_q) - 1 - i])
            cx(qs[comptime(n_q) - 1 - i], qs[i])
            cx(qs[i], qs[comptime(n_q) - 1 - i])

        for i in range(comptime(n_q)):
            for j in range(i):
                phi = -pi / (2 ** (i - j))
                rz(qs[j], phi / 2)
                crz(qs[j], qs[i], phi)
            h(qs[i])

    @guppy
    def qft_prep_n(qs: array[qubit, comptime(n_q)], inpInt: int) -> None:
        """Prepares Fourier state corresponding to computational state |inpInt>."""
        for i in range(comptime(n_q)):
            h(qs[i])

        for j in range(comptime(n_q)):
            rz(qs[j], (2.0 * pi * inpInt) / (2 ** (comptime(n_q) - j)))

    @guppy
    def main_qft_bench() -> None:
        val = comptime(inp_int)
        qs = array(qubit() for _ in range(comptime(n_q)))

        qft_prep_n(qs, val)
        iqft_n(qs)

        result("c", measure_array(qs))

    return main_qft_bench


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
    shots_list = sim_result.collated_shots()
    total_shots = len(shots_list)
    print(f"\n--- {label}: {total_shots} shots ---")
    
    counts = Counter()
    for shot in shots_list:
        bits = shot.get("c", [[]])[0]
        bitstr = "".join(str(int(b)) for b in reversed(bits))
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
    n_q = 7
    inp_int = 23

    # 1. Check Guppy Program
    main_qft_bench = get_main_bench(n_q, inp_int)
    print("\nChecking Guppy program...")
    print("  main_qft_bench check:", main_qft_bench.check())


    if n_q <5:
        circQi=guppy_to_qiskit(main_qft_bench,nq=n_q)
        print(circQi)
   
    # 3. Execution
    num_shots = 200
    if n_q < 23:
        print(f"\nRunning ideal simu locally (inpInt={inp_int}, nq={n_q}, shots={num_shots}) ...")
    
        # Ideal
        t0 = time()
        ideal_result = run_ideal(main_qft_bench, nq=n_q, shots=num_shots)
        elaT = time() - t0
        print(f"ideal simu elaT={elaT:.1f} sec")
        postprocess_shots(ideal_result, n_q, inp_int, label="Ideal")

    
    # Helios Emulator
    
    print("\nCompiling to HUGR IR for Helios...")
    hugr_pkg = main_qft_bench.compile()
    
    print(f"Submitting {num_shots} shots to Helios-1E...")
    ref_exec, tag = submit_job(hugr_pkg, nq=n_q, shots=num_shots, dev_name="Helios-1E")
    
    result_data = retrieve_job(ref_exec, nq=n_q)
    if result_data:
        postprocess_shots(result_data, n_q, inp_int, label="HeliosEmu")

if __name__ == "__main__":
    main()
    os._exit(0)  # Bypass native teardown to prevent segfaults from qnexus/pytket on exit
