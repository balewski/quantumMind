#!/usr/bin/env python3
__author__ = "Jan Balewski (Z-basis teleportation post-processing)"
__email__ = "janstar1122@gmail.com"

"""
Teleportation without feed-forward, using post-processing corrections

Teleportation in Z basis without quantum feed-forward.

We measure all 3 qubits:
  m_z = Alice's Z-basis outcome (qubit 0)
  m_x = Alice's X-basis outcome (qubit 1, via H+Z measurement)
  b_raw = Bob's raw Z-basis outcome (qubit 2)

Post-processing correction for Z-basis:
  - X correction (m_x = 1) flips Bob's bit.
  - Z correction (m_z = 1) has no effect in Z basis.

Truth table (Flip? = 1 means flip Bob's bit):

m_z | m_x | Flip?
----+-----+------
 0  |  0  |  0
 0  |  1  |  1
 1  |  0  |  0
 1  |  1  |  1
"""

import argparse
import numpy as np
import qiskit as qk
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler


# ----------------------------
# Teleportation circuit without feed-forward (Z basis only)
# ----------------------------
def circTeleport_noFF(secretState):
    qr = qk.QuantumRegister(3, name="q")
    cr = qk.ClassicalRegister(3, name="c")  # measure all 3 qubits
    qc = qk.QuantumCircuit(qr, cr, name="Teleport_noFF_Z")

    qs, qa, qb = 0, 1, 2

    # Prepare secret state on qubit 0
    qc.u(secretState[0], secretState[1], 0.0, qs)
    qc.barrier()

    # Create Bell pair between qubits 1 and 2
    qc.h(qa)
    qc.cx(qa, qb)
    qc.barrier()

    # Alice's Bell measurement
    qc.cx(qs, qa)
    qc.h(qs)

    # Z-basis measurement: no rotation on Bob's qubit

    # Measure all qubits
    qc.measure(qs, cr[0])  # Alice Z outcome
    qc.measure(qa, cr[1])  # Alice X outcome
    qc.measure(qb, cr[2])  # Bob raw Z outcome

    return qc


# ----------------------------
# Alice's classical decoder
# ----------------------------
def alice_decoder(bitstring):
    # Qiskit bitstring order: c2 c1 c0
    m_z = int(bitstring[2])  # Alice Z
    m_x = int(bitstring[1])  # Alice X
    return m_z, m_x


# ----------------------------
# Bob's classical decoder (Z basis only)
# ----------------------------
def bob_decoder(bobRaw, m_z, m_x):
    # In Z basis: flip only if m_x == 1
    if m_x == 1:
        return bobRaw ^ 1
    return bobRaw


# ----------------------------
# Post-processing correction
# ----------------------------
def post_process_counts(counts):
    corrected_counts = {'0': 0, '1': 0}
    for bitstring, count in counts.items():
        m_z, m_x = alice_decoder(bitstring)
        bobRaw = int(bitstring[0])  # c2
        corrected_bob = bob_decoder(bobRaw, m_z, m_x)
        corrected_counts[str(corrected_bob)] += count
    return corrected_counts


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_results(counts, secretState, nSigThr=3):
    total = sum(counts.values())
    n0 = counts.get('0', 0)
    n1 = counts.get('1', 0)

    prob1 = n1 / total
    prob_err = np.sqrt(n0 * n1 / total) / total

    ev_meas = 1 - 2 * prob1
    ev_err = 2 * prob_err

    theta, _phi = secretState
    ev_true = np.cos(theta)

    dev = ev_meas - ev_true
    nSig = abs(dev) / ev_err

    print("\n--- Evaluation (Z basis) ---")
    print(f"Measured <Z> = {ev_meas:.3f} ± {ev_err:.3f}")
    print(f"True     <Z> = {ev_true:.3f}")
    print(f"Deviation = {dev:.3f} ({nSig:.1f} σ)")

    if nSig < nSigThr:
        print(f"PASS: |nSig| < {nSigThr}")
    else:
        print(f"FAIL: |nSig| >= {nSigThr}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z-basis teleportation with post-processing correction")
    parser.add_argument(
        "--secretState", type=float, nargs=2, default=[2.4, 0.6],
        help="Secret state parameters [theta, phi] for U3 gate"
    )
    parser.add_argument(
        "--shots", type=int, default=2000,
        help="Number of shots for simulation"
    )
    args = parser.parse_args()

    # Build circuit
    qcTele = circTeleport_noFF(args.secretState)

    # Run on Aer simulator
    backend = AerSimulator()
    sampler = Sampler(mode=backend)
    job = sampler.run([qcTele], shots=args.shots)
    result = job.result()
    counts_raw = result[0].data.c.get_counts()

    print(qcTele.draw(output="text", idle_wires=False))
    print("Raw counts (c2 c1 c0):", counts_raw)

    # Post-process correction
    counts_corrected = post_process_counts(counts_raw)
    print("Corrected Bob counts:", counts_corrected)

    # Evaluate
    evaluate_results(counts_corrected, args.secretState, nSigThr=3)
