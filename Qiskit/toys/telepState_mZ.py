#!/usr/bin/env python3
__author__ = "Jan Balewski (simplified for Qiskit 2.x, no Z correction)"
__email__ = "janstar1122@gmail.com"

"""
Simple teleportation protocol with Z-basis measurement and feed-forward

Minimal Teleportation with tomography in Z basis
Runs on Aer simulator only
Evaluates measurement results with nSig threshold
Feed-forward Z-correction removed, because in Z-basis it makes no difference
"""

import argparse
import numpy as np
import qiskit as qk
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler


# ----------------------------
# Tomography measurement
# ----------------------------
def meas_tomo(qc, tq, tb, axMeas):
    """Measure qubit tq into classical bit tb in given basis (x, y, z)."""
    assert axMeas in 'xyz'
    if axMeas == 'y':
        qc.sx(tq)
    if axMeas == 'x':
        qc.rz(np.pi / 2, tq)
        qc.sx(tq)
        qc.rz(-np.pi / 2, tq)
    qc.measure(tq, tb)
    qc.name += '_' + axMeas


# ----------------------------
# Alice sends
# ----------------------------
def alice_send(qc, qs, qa, crz, crx):
    qc.cx(qs, qa)
    qc.h(qs)
    qc.measure(qs, crz[0])
    qc.measure(qa, crx[0])
    qc.barrier()


# ----------------------------
# Bob applies gates conditionally
# ----------------------------
def bob_gates_noZ(qc, qubit, crz, crx):
    """Only apply X correction — Z correction removed."""
    with qc.if_test((crx[0], 1)):
        qc.x(qubit)
    # No Z correction here


# ----------------------------
# Teleportation circuit
# ----------------------------
def circTeleport(axMeas, secretState):
    """
    axMeas: 'x', 'y', or 'z' — tomography basis for Bob's qubit
    secretState: [theta, phi] for U3 gate
    """
    qr = qk.QuantumRegister(3, name="q")
    crz = qk.ClassicalRegister(1, name="mz_alice")
    crx = qk.ClassicalRegister(1, name="mx_alice")
    crb = qk.ClassicalRegister(1, name="bob")
    qc = qk.QuantumCircuit(qr, crz, crx, crb, name="Teleport")

    qs, qa, qb = 0, 1, 2

    # Prepare secret state on qubit 0
    qc.u(secretState[0], secretState[1], 0.0, qs)
    qc.barrier()

    # Create Bell pair between qubits 1 and 2
    qc.h(qa)
    qc.cx(qa, qb)
    qc.barrier()

    # Alice's measurement and sending
    alice_send(qc, qs, qa, crz, crx)

    # Bob's conditional gates (only X correction)
    bob_gates_noZ(qc, qb, crz, crx)

    # Tomography measurement on Bob's qubit
    meas_tomo(qc, qb, crb[0], axMeas)

    return qc


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_results(counts, secretState, nSigThr=3):
    """Compare measured expectation value with theoretical prediction."""
    total = sum(counts.values())
    n0 = sum(v for k, v in counts.items() if k[0] == '0')
    n1 = total - n0

    # Measured probability of |1>
    prob1 = n1 / total
    prob_err = np.sqrt(n0 * n1 / total) / total

    # Measured expectation value <Z> = 1 - 2*P(1)
    ev_meas = 1 - 2 * prob1
    ev_err = 2 * prob_err

    # Theoretical expectation value for Z basis
    theta, phi = secretState
    ev_true = np.cos(theta)  # <Z> for |ψ> = U3(theta, phi, 0)|0>

    # Compare
    dev = ev_meas - ev_true
    nSig = abs(dev) / ev_err

    print("\n--- Evaluation ---")
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
    parser = argparse.ArgumentParser(description="Minimal teleportation in Z basis with evaluation (no Z correction)")
    parser.add_argument(
        "--secretState", type=float, nargs=2, default=[2.4, 0.6],
        help="Secret state parameters [theta, phi] for U3 gate"
    )
    parser.add_argument(
        "--shots", type=int, default=2000,
        help="Number of shots for simulation"
    )
    args = parser.parse_args()

    # Create the teleportation circuit with Z-basis tomography
    qcTele = circTeleport('z', args.secretState)

    # Run on Aer simulator
    backend = AerSimulator()
    sampler = Sampler(mode=backend)
    job = sampler.run([qcTele], shots=args.shots)
    result = job.result()
    counts = result[0].data.meas.get_counts()

    print(qcTele.draw(output="text", idle_wires=False))
    print("Counts:", counts)

    # Evaluate
    evaluate_results(counts, args.secretState, nSigThr=3)
