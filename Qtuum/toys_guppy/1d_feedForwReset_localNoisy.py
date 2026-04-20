#!/usr/bin/env python3
"""
Feed-forward and reset demo with noisy local simulation (no Nexus, no hardware).

Circuit: 
  - Prepare a Bell pair on q0,q1 (H + CX)
  - Mid-circuit measure q0
  - Conditionally apply X on q2 and q3 based on q0
  - Reset q3
  - Measure q1, q2, q3, and q4.

Pipeline:
  1. Define the circuit as a @guppy function (my_circ_fn)
  2. Run ideal stabilizer emulation as a baseline
  3. Run noisy statevector emulation with DepolarizingErrorModel
  4. Compare ideal vs noisy shot results
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from collections import Counter
from pprint import pprint

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, h, measure, qubit, x, reset
from selene_sim.backends.bundled_error_models import DepolarizingErrorModel
from toolbox.Util_Guppy import guppy_to_qiskit
from toolbox.Util_QiskitV2 import draw_circuit_png

# ---- Guppy circuit definitions ----

@guppy
def my_circ_fn() -> None:
    """Prepare Bell pair on q0,q1, feed-forward from q0 to q2,q3, reset q3, measure remaining."""
    q0, q1, q2, q3, q4 = qubit(), qubit(), qubit(), qubit(), qubit()
    
    # 1. Bell pair
    h(q0)
    cx(q0, q1)
    
    # 2. Mid-circuit measure q0 (trigger)
    outcome = measure(q0)
    result("q0", outcome)
    
    # 3. Feed-forward to q2 and q3
    if outcome:
        x(q2)
        x(q3)
        
    # 4. Reset q3
    reset(q3)
    
    # 5. Measure the rest
    result("q1", measure(q1))
    result("q2", measure(q2))
    result("q3", measure(q3))
    result("q4", measure(q4))


def build_error_model(p_1q=0.001, p_2q=0.01, p_meas=0.001, p_init=0.001,
                      random_seed=None):
    """Configure a depolarizing error model for noisy simulation."""
    model = DepolarizingErrorModel(
        p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, p_init=p_init,
        random_seed=random_seed,
    )
    print(f"\nError model: DepolarizingErrorModel"
          f"(p_1q={p_1q}, p_2q={p_2q}, p_meas={p_meas}, p_init={p_init})")
    return model


def run_ideal(guppy_prog, n_qubits: int = 5,
              shots: int = 5, seed: int = 42):
    """Run local stabilizer emulation (ideal, no noise)."""
    emulator = guppy_prog.emulator(n_qubits=n_qubits).stabilizer_sim().with_seed(seed)
    sim_result = emulator.with_shots(shots).run()
    return sim_result


def run_noisy(guppy_prog, error_model, n_qubits: int = 5,
              shots: int = 100, seed: int = 42):
    """Run local statevector emulation with a noise model."""
    emulator = (guppy_prog.emulator(n_qubits=n_qubits)
                .with_error_model(error_model)
                .with_seed(seed))
    sim_result = emulator.with_shots(shots).run()
    return sim_result


def postprocess_shots(sim_result, label=""):
    """Print per-shot entries and aggregated counts."""
    shots = sim_result.results
    meas_names = [name for name, _ in shots[0].entries]
    print(f"\n--- {label}: {len(shots)} shots ---")
    for i, shot in enumerate(shots):
        entries = {name: val for name, val in shot.entries}
        # Convert True/False to 1/0 for cleaner strings
        bitstr = "".join(str(int(entries[name])) for name in meas_names)
        print(f"  shot {i}: {bitstr}")
        if i>6:  # limit per-shot printout
            print("  ...")
            break   

    # Aggregate counts
    counts = Counter()
    for shot in shots:
        entries = {name: val for name, val in shot.entries}
        bitstr = "".join(str(int(entries[name])) for name in meas_names)
        counts[bitstr] += 1
    print(f"\n{label} counts:")
    qubit_order = [int(name[1:]) for name in meas_names]
    print(f"  qubit order {qubit_order}")
    for bitstr, cnt in counts.most_common():
        print(f"  {bitstr}:{cnt}")


# ---- main ----

def main():
    qcQi=guppy_to_qiskit(my_circ_fn,nq=5)
    draw_circuit_png(qcQi,outName='out/1d_feedForwReset.png',title='Guppy FF+Reset')
    print(qcQi)
    print("check:", my_circ_fn.check())
    
    shots = 1000
    
    # Ideal stabilizer emulation (baseline)
    ideal_result = run_ideal(my_circ_fn, n_qubits=5, shots=shots)
    postprocess_shots(ideal_result, label="Ideal")

    # Noisy statevector emulation
    error_model = build_error_model(p_1q=0.001, p_2q=0.01,
                                    p_meas=0.001, p_init=0.001)
    noisy_result = run_noisy(my_circ_fn, error_model, n_qubits=5, shots=shots)
    postprocess_shots(noisy_result, label="Noisy")


if __name__ == "__main__":
    main()
