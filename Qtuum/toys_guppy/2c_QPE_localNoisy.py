#!/usr/bin/env python3
"""
Canonical Quantum Phase estimation with local noisy simulation.
Based on: https://docs.quantinuum.com/guppy/guppylang/examples/canonical-qpe.html

We will consider a very simplified version of phase estimation wherein 
 is a diagonal matrix. This means the true eigenvalues can be read off the diagonal.
  This will allow us to clearly see that our implementation is correct.

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
from collections import Counter

from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.quantum import qubit, h, crz, x, measure_array, discard_array
from guppylang.std.builtins import result, array, comptime
from guppylang.std.mem import mem_swap

from pytket.circuit import DiagonalBox, QControlBox
from pytket.passes import AutoRebase
from pytket import OpType

from selene_sim.backends.bundled_error_models import DepolarizingErrorModel
from toolbox.Util_Guppy import guppy_to_qiskit

# ---- Pytket / Guppy integration for Controlled-U ----

def setup_controlled_u():
    """Initializes the controlled-U gate by rebasing a Pytket DiagonalBox."""
    # Define the unitary U (diagonal)
    # For eigenstate |11>, eigenvalue is exp(i * pi / 8)
    d_box = DiagonalBox(np.array([1, 1, np.exp(1j * np.pi / 4), np.exp(1j * np.pi / 8)]))
    controlled_u_op = QControlBox(d_box, 1)

    # Rebase for Guppy compatibility
    rebase = AutoRebase({OpType.CX, OpType.Rz, OpType.H, OpType.CCX})
    circ = controlled_u_op.get_circuit()
    rebase.apply(circ)

    # Load into Guppy
    return guppy.load_pytket("controlled_u_circuit", circ, use_arrays=False)

# Global handle for the gate used in Guppy subroutines
controlled_u = setup_controlled_u()

# ---- Guppy circuit definitions ----

@guppy
def prepare_trivial_eigenstate() -> array[qubit, 2]:
    """Prepares the eigenstate |11>."""
    q0, q1 = qubit(), qubit()
    x(q0)
    x(q1)
    return array(q0, q1)

def get_main_bench(m_q: int):
    """Returns a Guppy program specialized to the requested measurement-register size."""

    @guppy
    def inverse_qft(qs: array[qubit, comptime(m_q)]) -> None:
        """General Inverse QFT on m_q qubits."""
        for k in range(comptime(m_q // 2)):
            mem_swap(qs[k], qs[comptime(m_q) - k - 1])
        for i in range(comptime(m_q)):
            h(qs[comptime(m_q) - i - 1])
            for j in range(comptime(m_q) - i - 1):
                crz(qs[comptime(m_q) - i - 1], qs[comptime(m_q) - i - j - 2], -pi / (2 ** (j + 1)))

    @guppy
    def phase_estimation(measured: array[qubit, comptime(m_q)], state: array[qubit, 2]) -> None:
        """Standard QPE algorithm."""
        for i in range(comptime(m_q)):
            h(measured[i])

        for n_index in range(comptime(m_q)):
            control_index: int = comptime(m_q) - n_index - 1
            for _ in range(2**n_index):
                controlled_u(measured[control_index], state[0], state[1])

        inverse_qft(measured)

    @guppy
    def main_qpe_bench() -> None:
        state = prepare_trivial_eigenstate()
        measured = array(qubit() for _ in range(comptime(m_q)))

        phase_estimation(measured, state)
        discard_array(state)
        result("c", measure_array(measured))

    return main_qpe_bench

# ---- Helpers for simulation ----

def build_error_model(p_1q=0.001, p_2q=0.01, p_meas=0.001, p_init=0.001, seed=None):
    model = DepolarizingErrorModel(
        p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, p_init=p_init,
        random_seed=seed,
    )
    print(f"Error model: p_1q={p_1q}, p_2q={p_2q}, p_meas={p_meas}")
    return model

def run_ideal(guppy_prog, n_qubits, shots, seed=42):
    emulator = guppy_prog.emulator(n_qubits=n_qubits).statevector_sim().with_seed(seed)
    return emulator.with_shots(shots).run()

def run_noisy(guppy_prog, error_model, n_qubits, shots, seed=42):
    emulator = (guppy_prog.emulator(n_qubits=n_qubits)
                .with_error_model(error_model)
                .with_seed(seed))
    return emulator.with_shots(shots).run()

def postprocess_qpe(sim_result, nq, correct_bitstr="0001", label=""):
    shots_list = sim_result.collated_shots()
    total = len(shots_list)
    print(f"\n--- {label}: {total} shots ---")
    
    counts = Counter()
    for shot in shots_list:
        bits = shot.get("c", [[]])[0]
        bitstr = "".join(str(int(b)) for b in bits)
        counts[bitstr] += 1
    
    # Calculate total success across the entire dataset
    success = counts.get(correct_bitstr, 0)
    
    # Top results for display
    for bitstr, cnt in counts.most_common(6):
        val = int(bitstr, 2)
        print(f"  {bitstr} (dec={val:2d}): {cnt}")
        
    # Stats like QFT
    prob = success / total if total > 0 else 0
    err = np.sqrt(prob * (1 - prob) / total) if total > 0 else 0
    print(f"QPE nq_meas:{nq}  prob_success: {prob:.4f} +/- {err:.4f}, Success: {success}")
    return prob

# ---- Main Execution ----

def main():
    verb=1  # verbosity
    mq = 5  # mental/measurement qubits    WARN: must match measured = array(qubit(),...) in main_qpe_bench() 
    sq = 2  # state qubits
    total_q = mq + sq
    correct_bitstr = "0001" + "0" * (mq - 4)
    main_qpe_bench = get_main_bench(mq)
    

    print(f"Checking Guppy QPE program... mq={mq}, sq={sq}, total_q={total_q}, correct_bitstr={correct_bitstr}")
    print("  main_qpe_bench check:", main_qpe_bench.check())
    
    # 1a. Visualization
    circQi = guppy_to_qiskit(main_qpe_bench, nq=total_q)
    print("\nGuppy QPE Circuit (via Qiskit):")
    if verb>1:
        print(circQi)
    print("\nGate counts:", circQi.count_ops())

    num_shots = 500
    
    # 1. Ideal Simulation
    print(f"\nRunning ideal QPE (mq={mq}, sq={sq}, total={total_q})")
    ideal_res = run_ideal(main_qpe_bench, n_qubits=total_q, shots=num_shots)
    p_ideal = postprocess_qpe(ideal_res, mq, correct_bitstr=correct_bitstr, label="Ideal")
    
    # 2. Noisy Simulation
    print(f"\nRunning noisy QPE...")
    error_model = build_error_model(p_1q=0.001, p_2q=0.001, p_meas=0.001)
    noisy_res = run_noisy(main_qpe_bench, error_model, n_qubits=total_q, shots=num_shots)
    p_noisy = postprocess_qpe(noisy_res, mq, correct_bitstr=correct_bitstr, label="Noisy")
    
    # 3. Fidelity Comparison
    # As an "evaluation function", we compare the success probability
    fidelity_ratio = p_noisy / p_ideal if p_ideal > 0 else 0
    print(f"\nFinal Comparison:")
    print(f"  Ideal success probability: {p_ideal:.4f}")
    print(f"  Noisy success probability: {p_noisy:.4f}")
    print(f"  Relative Fidelity (Noisy/Ideal): {fidelity_ratio:.4f}")

if __name__ == "__main__":
    import os
    main()
    os._exit(0)
