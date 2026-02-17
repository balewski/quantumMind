#!/usr/bin/env python3
"""
Feed-forward demo with noisy local simulation (no Nexus, no hardware).

Circuit: prepare a Bell pair (H + CX), mid-circuit measure q1,
conditionally apply X on q2 to disentangle it → q2 always collapses to |0⟩.

Pipeline:
  1. Define the circuit as a @guppy function (my_circ_fn)
  2. Wrap it in an evaluator that measures the returned qubit
  3. Run ideal stabilizer emulation as a baseline
  4. Run noisy statevector emulation with DepolarizingErrorModel
  5. Compare ideal vs noisy shot results

Ref:  https://docs.quantinuum.com/guppy/api/emulator.html
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from collections import Counter
from pprint import pprint

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, h, measure, qubit, x
from selene_sim.backends.bundled_error_models import DepolarizingErrorModel
from toolbox.Util_Guppy import guppy_to_qiskit
from toolbox.Util_QiskitV2 import draw_circuit_png

# ---- Guppy circuit definitions ----

@guppy
def my_circ_fn() -> qubit:
    """Prepare Bell pair, measure q1, conditionally flip q2."""
    q1, q2 = qubit(), qubit()
    h(q1)
    cx(q1, q2)
    outcome = measure(q1)
    result("q1", outcome)
    if outcome:
        x(q2)
    return q2


# ---- helpers ----

def build_evaluator(circuit_fn):
    """Build a guppy evaluator that measures the output qubit of *circuit_fn*."""
    @guppy
    def _evaluate() -> None:
        q = circuit_fn()
        result("q2", measure(q))
    return _evaluate


def build_error_model(p_1q=0.001, p_2q=0.01, p_meas=0.001, p_init=0.001,
                      random_seed=None):
    """Configure a depolarizing error model for noisy simulation.

    Args:
        p_1q:   depolarizing probability for 1-qubit gates
        p_2q:   depolarizing probability for 2-qubit gates
        p_meas: measurement error probability
        p_init: state-preparation error probability
        random_seed: optional seed for reproducibility
    Returns:
        DepolarizingErrorModel instance
    """
    model = DepolarizingErrorModel(
        p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, p_init=p_init,
        random_seed=random_seed,
    )
    print(f"Error model: DepolarizingErrorModel"
          f"(p_1q={p_1q}, p_2q={p_2q}, p_meas={p_meas}, p_init={p_init})")
    return model


def run_ideal(guppy_prog, n_qubits: int = 2,
              shots: int = 5, seed: int = 42):
    """Run local stabilizer emulation (ideal, no noise)."""
    emulator = guppy_prog.emulator(n_qubits=n_qubits).stabilizer_sim().with_seed(seed)
    sim_result = emulator.with_shots(shots).run()
    return sim_result


def run_noisy(guppy_prog, error_model, n_qubits: int = 2,
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
    print(f"\n--- {label}: {len(shots)} shots ---")
    for i, shot in enumerate(shots):
        entries = {name: val for name, val in shot.entries}
        print(f"  shot {i}: {entries}")

    # Aggregate counts
    counts = Counter()
    for shot in shots:
        key = tuple(f"{name}={val}" for name, val in shot.entries)
        counts[key] += 1
    print(f"\n{label} counts:")
    for key, cnt in counts.most_common():
        print(f'  {", ".join(key)}: {cnt}')



# ---- main ----

def main():
    qcQi=guppy_to_qiskit(my_circ_fn,nq=3)
    draw_circuit_png(qcQi,outName='out/1cfeed.png',title='first Gupy circ')
    print(qcQi)
    print("check:", my_circ_fn.check())
    
    shots = 10

    # Build evaluator for the chosen circuit
    prog1 = build_evaluator(my_circ_fn)

    # Ideal stabilizer emulation (baseline)
    ideal_result = run_ideal(prog1, n_qubits=2, shots=shots)
    postprocess_shots(ideal_result, label="Ideal")

    # Noisy statevector emulation
    error_model = build_error_model(p_1q=0.001, p_2q=0.01,
                                    p_meas=0.001, p_init=0.001)
    noisy_result = run_noisy(prog1, error_model, n_qubits=2, shots=shots)
    postprocess_shots(noisy_result, label="Noisy")


if __name__ == "__main__":
    main()
