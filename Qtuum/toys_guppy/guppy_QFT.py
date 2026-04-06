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
from guppylang.std.builtins import result, comptime
from guppylang.std.quantum import h, rz, crz, cx, measure, measure_array, qubit, pi, array
from guppylang.std.mem import mem_swap

from selene_sim.backends.bundled_error_models import DepolarizingErrorModel
from toolbox.Util_Guppy import guppy_to_qiskit

# ---- Guppy circuit definitions ----

def get_main_bench(n_q: int, inp_int: int):
    """Returns a Guppy program configured for the specific nq using a factory pattern.
    Uses 'comptime' to specialize all inner functions to the given qubit count and input state.
    """
    
    @guppy
    def iqft_n(qs: array[qubit, comptime(n_q)]) -> None:
        """Standard Inverse QFT on n_q qubits matching Qiskit's QFTGate.inverse()"""
        # 1. Reverse qubit order with physical SWAP gates
        for i in range(comptime(n_q // 2)):
            cx(qs[i], qs[comptime(n_q) - 1 - i])
            cx(qs[comptime(n_q) - 1 - i], qs[i])
            cx(qs[i], qs[comptime(n_q) - 1 - i])
        
        # 2. Iterate through qubits and apply rotations and H gates
        for i in range(comptime(n_q)):
            # Apply managed rotations
            for j in range(i):
                phi = -pi / (2 ** (i - j))
                rz(qs[j], phi / 2)
                crz(qs[j], qs[i], phi)
            h(qs[i])

    @guppy
    def qft_prep_n(qs: array[qubit, comptime(n_q)], inpInt: int) -> None:
        """Prepares Fourier state corresponding to computational state |inpInt|."""
        # 1. Apply H on all
        for i in range(comptime(n_q)):
            h(qs[i])
        
        # 2. Apply relative phases based on inpInt
        for j in range(comptime(n_q)):
            rz(qs[j], (2.0 * pi * inpInt) / (2 ** (comptime(n_q) - j)))

    @guppy
    def main_qft_bench() -> None:
        val = comptime(inp_int)
        # Allocate array of n_q qubits using comptime
        qs = array(qubit() for _ in range(comptime(n_q)))
        
        qft_prep_n(qs, val)
        iqft_n(qs)
        
        # Measure entire array into a single result tag "c"
        result("c", measure_array(qs))
        
    return main_qft_bench


# ---- Helpers for simulation ----
 

# ---- Helpers for simulation ----

def build_error_model(p_1q=0.001, p_2q=0.01, p_meas=0.001, p_init=0.001, seed=None):
    model = DepolarizingErrorModel(
        p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, p_init=p_init,
        random_seed=seed,
    )
    print(f"Error model: DepolarizingErrorModel(p_1q={p_1q}, p_2q={p_2q}, p_meas={p_meas}, p_init={p_init})")
    return model

def run_ideal(guppy_prog, nq, shots, seed=42):
    # QFT requires statevector_sim because of arbitrary rotations.
    emulator = guppy_prog.emulator(n_qubits=nq).statevector_sim().with_seed(seed)
    return emulator.with_shots(shots).run()

def run_noisy(guppy_prog, error_model, nq, shots, seed=42):
    emulator = (guppy_prog.emulator(n_qubits=nq)
                .with_error_model(error_model)
                .with_seed(seed))
    return emulator.with_shots(shots).run()

def postprocess_shots(sim_result, nq, correct_int, label=""):
    shots_list = sim_result.collated_shots()
    total_shots = len(shots_list)
    print(f"\n--- {label}: {total_shots} shots ---")
    
    counts = Counter()
    for shot in shots_list:
        # Get bits from result "c" - typically nested as [[b0, b1, ...]]
        bits = shot.get("c", [[]])[0]
        # Construct bitstring (MSB on the left)
        bitstr = "".join(str(int(b)) for b in reversed(bits))
        counts[bitstr] += 1
    
    success_shots = 0
    # Sort and print
    for bitstr, cnt in counts.most_common():
        val = int(bitstr, 2)
        if val == correct_int:
            success_shots += cnt
        print(f"  {bitstr} (dec={val:2d}): {cnt}")

    if total_shots > 0:
        incorrect_shots = total_shots - success_shots
        prob = success_shots / total_shots
        err_prob = incorrect_shots / total_shots
        
        incorrect_states_seen = sum(1 for bitstr in counts if int(bitstr, 2) != correct_int)
        total_states_space = 2**nq
        err_err = np.sqrt(prob * (1 - prob) / total_shots)
        
        print(f"\nIncorrect states: {incorrect_states_seen} of {total_states_space}, total probability: {err_prob:.4f} ({incorrect_shots}/{total_shots})")
        print(f"QFT nq:{nq}   prob: {prob:.4f} +/- {err_err:.4f},   Success: {success_shots}")

# ---- Main Execution ----

def main():
    n_q = 7
    inp_int = 23

    # 1. Check Guppy Program
    main_qft_bench = get_main_bench(n_q, inp_int)
    print("\nChecking Guppy program...")
    print("  main_qft_bench check:", main_qft_bench.check())

    if n_q < 5:
        circQi = guppy_to_qiskit(main_qft_bench, nq=n_q)
        print(circQi)

    # 3. Execution
    num_shots = 100
    print(f"\nRunning simulations (inpInt={inp_int}, nq={n_q}, shots={num_shots}) ...")

    # Ideal
    ideal_result = run_ideal(main_qft_bench, nq=n_q, shots=num_shots)
    postprocess_shots(ideal_result, n_q, inp_int, label="Ideal")

    # Noisy
    error_model = build_error_model(p_1q=0.001, p_2q=0.01, p_meas=0.01)
    noisy_result = run_noisy(main_qft_bench, error_model, nq=n_q, shots=num_shots)
    postprocess_shots(noisy_result, n_q, inp_int, label="Noisy")

if __name__ == "__main__":
    main()
