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

from selene_sim.backends.bundled_error_models import DepolarizingErrorModel
from toolbox.Util_Guppy import guppy_to_qiskit

# ---- Guppy circuit definitions ----

INP_INT =5
NQ_VAL = 4
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
    """Returns a Guppy program configured for the specific nq using if/else statements.
    NOTE: Because Guppy does not capture Python variables (globals or closures), and the
    entry-point must have no parameters, the parameters must be hardcoded inside.
    """
    if nq == 2:
        @guppy
        def main_qft_bench() -> None:
            inpInt = comptime(INP_INT)
            qs = array(qubit(), qubit())
            qft_prep_n(qs, inpInt)
            iqft_n(qs)
            bits = measure_array(qs)
            result("b0", bits[0])
            result("b1", bits[1])
        return main_qft_bench

    elif nq == 3:
        @guppy
        def main_qft_bench() -> None:
            inpInt = comptime(INP_INT)
            qs = array(qubit(), qubit(), qubit())
            qft_prep_n(qs, inpInt)
            iqft_n(qs)
            bits = measure_array(qs)
            result("b0", bits[0])
            result("b1", bits[1])
            result("b2", bits[2])
        return main_qft_bench

    elif nq == 4:
        @guppy
        def main_qft_bench() -> None:
            inpInt = comptime(INP_INT)
            qs = array(qubit(), qubit(), qubit(), qubit())
            qft_prep_n(qs, inpInt)
            iqft_n(qs)
            bits = measure_array(qs)
            result("b0", bits[0])
            result("b1", bits[1])
            result("b2", bits[2])
            result("b3", bits[3])
        return main_qft_bench

    elif nq == 5:
        @guppy
        def main_qft_bench() -> None:
            inpInt = comptime(INP_INT)
            qs = array(qubit(), qubit(), qubit(), qubit(), qubit())
            qft_prep_n(qs, inpInt)
            iqft_n(qs)
            bits = measure_array(qs)
            result("b0", bits[0])
            result("b1", bits[1])
            result("b2", bits[2])
            result("b3", bits[3])
            result("b4", bits[4])
        return main_qft_bench

    elif nq == 6:
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
    else:
        raise ValueError(f"nq={nq} is not supported. Must be between 2 and 6 inclusive.")


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

def postprocess_shots(sim_result, nq, label=""):
    shots = sim_result.results
    print(f"\n--- {label}: {len(shots)} shots ---")
    
    counts = Counter()
    for shot in shots:
        entries = {name: val for name, val in shot.entries}
        # Construct bitstring. Qiskit style: MSB on the left -> b5 b4 b3 b2 b1 b0
        bitstr = "".join(str(entries[f"b{i}"]) for i in reversed(range(nq)))
        counts[bitstr] += 1
    
    # Sort and print
    for bitstr, cnt in counts.most_common():
        val = int(bitstr, 2)
        print(f"  {bitstr} (dec={val:2d}): {cnt}")

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
    num_shots = 100
    print(f"\nRunning simulations (inpInt={INP_INT}, nq={NQ_VAL}, shots={num_shots}) ...")
    
    # Ideal
    ideal_result = run_ideal(main_qft_bench, nq=NQ_VAL, shots=num_shots)
    postprocess_shots(ideal_result, NQ_VAL, label="Ideal")

    
    # Noisy
    error_model = build_error_model(p_1q=0.001, p_2q=0.01, p_meas=0.01)
    noisy_result = run_noisy(main_qft_bench, error_model, nq=NQ_VAL, shots=num_shots)
    postprocess_shots(noisy_result, NQ_VAL, label="Noisy")

if __name__ == "__main__":
    main()
