#!/usr/bin/env python3
__author__ = "Jan Balewski (extended for Qiskit 2.x, with full feed-forward)"
__email__ = "janstar1122@gmail.com"

"""
Minimal Teleportation with tomography for X, Y, Z bases
Runs on Aer simulator only  
Evaluates measurement results with nSig threshold
Includes proper feed-forward with both Z and X corrections
Works for any choice of measurement basis (x, y, z)
"""

import argparse
import numpy as np
import qiskit as qk
from qiskit_aer import Aer


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
    

# ----------------------------
# Bob applies gates conditionally
# ----------------------------
def bob_gates_with_feedforward(qc, qubit, crz, crx):
    """Apply both Z and X corrections via feed-forward."""
    with qc.if_test((crz[0], 1)):
        qc.z(qubit)
    with qc.if_test((crx[0], 1)):
        qc.x(qubit)


# ----------------------------
# Teleportation circuit
# ----------------------------
def circTeleport(axMeas, secretState, delayLine=0):
    """
    axMeas: 'x', 'y', or 'z' — tomography basis for Bob's qubit
    secretState: [theta, phi] for U3 gate
    delayLine: number of delay qubits for Bob's resource
    """
    # Total qubits: 3 + delayLine (secret, alice_bell, bob_bell, delay_qubits...)
    nQubits = 3 + delayLine
    qr = qk.QuantumRegister(nQubits, name="q")
    crz = qk.ClassicalRegister(1, name="mz_alice")
    crx = qk.ClassicalRegister(1, name="mx_alice")
    crb = qk.ClassicalRegister(1, name="bob")
    qc = qk.QuantumCircuit(qr, crz, crx, crb, name="Teleport")

    qs, qa, qb = 0, 1, 2  # secret, alice_bell, bob_bell
    qb_final = qb + delayLine  # final qubit where Bob measures

    
    # Create Bell pair between qubits 1 and 2
    qc.h(qa)
    qc.cx(qa, qb)
    qc.barrier()
    
    # Prepare secret state on qubit 0
    qc.ry(secretState[0],qs)
    qc.rz(secretState[1],qs)
    

    # Alice's measurement and sending
    alice_send(qc, qs, qa, crz, crx)

    
    # Delay line: sequentially swap Bob's qubit through delay qubits
    if delayLine > 0:
        qc.barrier()
        for i in range(delayLine):
            qc.swap(qb + i, qb + i + 1)
    

    # Bob's conditional gates (both Z and X corrections) applied to final qubit
    bob_gates_with_feedforward(qc, qb_final, crz, crx)

    # Tomography measurement on Bob's final qubit
    meas_tomo(qc, qb_final, crb[0], axMeas)

    return qc


# ----------------------------
# Theoretical expectation values
# ----------------------------
def theoretical_expectation(secretState, basis):
    """Calculate theoretical expectation value for given basis."""
    theta, phi = secretState
    if basis == 'z':
        return np.cos(theta)
    elif basis == 'x':
        return np.sin(theta) * np.cos(phi)
    elif basis == 'y':
        return np.sin(theta) * np.sin(phi)
    else:
        raise ValueError("Invalid basis")


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_results(counts, secretState, basis, nSigThr=3):
    """Compare measured expectation value with theoretical prediction."""
    total = sum(counts.values())
    n0 = sum(v for k, v in counts.items() if k[0] == '0')
    n1 = total - n0

    # Measured probability of |1>
    prob1 = n1 / total
    prob_err = np.sqrt(n0 * n1 / total) / total

    # Measured expectation value <B> = 1 - 2*P(1) where B is the measurement basis
    ev_meas = 1 - 2 * prob1
    ev_err = 2 * prob_err

    # Theoretical expectation value for given basis
    ev_true = theoretical_expectation(secretState, basis)

    # Compare
    dev = ev_meas - ev_true
    nSig = abs(dev) / ev_err

    print("\n--- Evaluation ---")
    print(f"Basis = {basis.upper()}")
    print(f"Measured <{basis.upper()}> = {ev_meas:.3f} ± {ev_err:.3f}")
    print(f"True     <{basis.upper()}> = {ev_true:.3f}")
    print(f"Deviation = {dev:.3f} ({nSig:.1f} σ)")

    if nSig < nSigThr:
        print(f"PASS: |nSig| < {nSigThr}")
    else:
        print(f"FAIL: |nSig| >= {nSigThr}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal teleportation with tomography and feed-forward for all bases")
    parser.add_argument(
        '-s',"--secretState", type=float, nargs=2, default=[2.4, 0.6],
        help="Secret state parameters [theta, phi] for U3 gate"
    )
    parser.add_argument(
        '-n',"--shots", type=int, default=20_000,
        help="Number of shots for simulation"
    )
    parser.add_argument(
        '-b', "--basis", type=str, choices=['x', 'y', 'z'], default='z',
        help="Tomography basis for Bob's qubit"
    )
    parser.add_argument(
        '-r',"--random", action='store_true', default=False,
        help="Generate random secret state (overrides --secretState)"
    )
    parser.add_argument(
        '-d',"--delayLine", type=int, default=0,
        help="Number of delay qubits for Bob's resource (default 0)"
    )
    args = parser.parse_args()

    # Generate random secret state if requested
    if args.random:
        # theta: polar angle [0, π], phi: azimuthal angle [0, 2π]
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        args.secretState = [theta, phi]
        print(f"Random secret state: theta={theta:.3f}, phi={phi:.3f}")
    else:
        print(f"Using specified secret state: theta={args.secretState[0]:.3f}, phi={args.secretState[1]:.3f}")
    print(f"Delay line length: {args.delayLine} qubits")

    # Create the teleportation circuit with specified basis tomography
    qcTele = circTeleport(args.basis, args.secretState, args.delayLine)

    # Run on Aer simulator
    backend = Aer.get_backend('aer_simulator')
    job = backend.run(qcTele, shots=args.shots)
    result = job.result()
    counts = result.get_counts(qcTele)

    print(qcTele.draw(output="text", idle_wires=False))
    print("Counts:", counts)

    # Evaluate
    evaluate_results(counts, args.secretState, args.basis, nSigThr=3)
