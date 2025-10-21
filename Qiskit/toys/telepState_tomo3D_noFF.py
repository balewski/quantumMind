#!/usr/bin/env python3
__author__ = "Jan Balewski (post-processing teleportation tomography)"
__email__ = "janstar1122@gmail.com"

"""
Teleportation tomography template with classical post-processing for all measurement bases

SUMMARY
=======
This script demonstrates quantum teleportation *without* quantum feed-forward.

1. **Quantum Part**:
   - Prepare a secret state on qubit 0: U3(theta, phi, 0) |0>
   - Create a Bell pair between qubits 1 and 2.
   - Perform teleportation steps (CNOT, H) on Alice's side.
   - Measure all 3 qubits in the computational basis, after optionally
     rotating Bob's qubit to measure in X, Y, or Z basis (tomography).

2. **Classical Part** (post-processing):
We measure all 3 qubits:
  m_z = Alice's Z‑basis outcome (qubit 0)
  m_x = Alice's X‑basis outcome (qubit 1, via H+Z measurement)
  b_raw = Bob's raw outcome in chosen basis (qubit 2)

Alice sends (m_z, m_x) to Bob.  
Bob’s classical correction depends on his measurement basis:

Decoding truth table (Flip? = 1 means flip Bob's bit in post‑processing):

Basis | m_z | m_x | Flip? | Notes
------+-----+-----+-------+-----------------------------
  Z   |  0  |  0  |   0   | no change
  Z   |  0  |  1  |   1   | X flips in Z basis
  Z   |  1  |  0  |   0   | Z no effect in Z basis
  Z   |  1  |  1  |   1   | X flips in Z basis
  X   |  0  |  0  |   0   | no change
  X   |  0  |  1  |   0   | X no effect in X basis
  X   |  1  |  0  |   1   | Z flips in X basis
  X   |  1  |  1  |   1   | Z flips in X basis
  Y   |  0  |  0  |   0   | no change
  Y   |  0  |  1  |   1   | X flips in Y basis
  Y   |  1  |  0  |   1   | Z flips in Y basis
  Y   |  1  |  1  |   0   | X and Z flip → cancel

This table is implemented in bob_decoder().

3. **Evaluation**:
   - Compute the measured expectation value <B> where B is Z, X, or Y.
   - Compare with theoretical expectation from the secret state.
   - Report deviation in units of standard error (nSig) and PASS/FAIL
     against a threshold (default nSigThr = 3).

Command-line arguments:
    --secretState theta phi   (default 2.4 0.6)
    --shots N                 (default 2000)
    --basis z|x|y              (default z)

Example:
    python3 teleport_postprocess_tomo.py --secretState 2.4 0.6 --shots 2000 --basis x
"""

import argparse
import numpy as np
import qiskit as qk
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler


# ----------------------------
# Teleportation circuit without feed-forward
# ----------------------------
def circTeleport_noFF(meas_basis, secretState):
    """
    meas_basis: 'x', 'y', or 'z' — tomography basis for Bob's qubit
    secretState: [theta, phi] for U3 gate
    """
    qr = qk.QuantumRegister(3, name="q")
    cr = qk.ClassicalRegister(3, name="c")  # measure all 3 qubits
    qc = qk.QuantumCircuit(qr, cr, name="Teleport_noFF")

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

    # Rotate Bob's qubit for tomography
    if meas_basis == 'y':
        qc.sx(qb)  # R_y(-pi/2) equivalent
    if meas_basis == 'x':
        qc.rz(np.pi / 2, qb)
        qc.sx(qb)
        qc.rz(-np.pi / 2, qb)

    # Measure all qubits
    qc.measure(qs, cr[0])  # Alice Z-meas bit
    qc.measure(qa, cr[1])  # Alice X-meas bit
    qc.measure(qb, cr[2])  # Bob's raw bit

    return qc


# ----------------------------
# Alice's classical decoder
# ----------------------------
def alice_decoder(bitstring):
    """
    Extract Alice's measurement results from raw bitstring.
    bitstring: 'c2 c1 c0' (Qiskit order: c[0]=rightmost char)
      c0 = AliceZ  (from qubit 0)
      c1 = AliceX  (from qubit 1)
      c2 = BobRaw  (from qubit 2)
    Returns: (aliceZ, aliceX)
    """
    aliceZ = int(bitstring[2])  # c0
    aliceX = int(bitstring[1])  # c1
    return aliceZ, aliceX


# ----------------------------
# Bob's classical decoder
# ----------------------------
def bob_decoder(bobRaw, m_z, m_x, basis='z'):
    """Apply correct classical correction in post-processing based on basis."""
    flip = 0
    if basis == 'z':
        if m_x == 1:
            flip = 1
    elif basis == 'x':
        if m_z == 1:
            flip = 1
    elif basis == 'y':
        # In Y basis, X and Z corrections flip in different cases
        if (m_x ^ m_z) == 1:  # XOR
            flip = 1
    else:
        raise ValueError("Invalid basis")
    if flip:
        return bobRaw ^ 1
    return bobRaw


# ----------------------------
# Post-processing pipeline
# ----------------------------
def post_process_counts(counts, meas_basis='z'):
    corrected_counts = {'0': 0, '1': 0}
    for bitstring, count in counts.items():
        aliceZ, aliceX = alice_decoder(bitstring)
        bobRaw = int(bitstring[0])  # c2
        corrected_bob = bob_decoder(bobRaw, aliceZ, aliceX, meas_basis)
        corrected_counts[str(corrected_bob)] += count
    return corrected_counts


# ----------------------------
# Theoretical expectation values
# ----------------------------
def theoretical_expectation(secretState, basis):
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
    n0 = counts.get('0', 0)
    n1 = counts.get('1', 0)

    prob1 = n1 / total
    prob_err = np.sqrt(n0 * n1 / total) / total

    ev_meas = 1 - 2 * prob1
    ev_err = 2 * prob_err

    ev_true = theoretical_expectation(secretState, basis)

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
    parser = argparse.ArgumentParser(description="Teleportation tomography with post-processing correction")
    parser.add_argument(
        "--secretState", type=float, nargs=2, default=[2.4, 0.6],
        help="Secret state parameters [theta, phi] for U3 gate"
    )
    parser.add_argument(
        "--shots", type=int, default=2000,
        help="Number of shots for simulation"
    )
    parser.add_argument(
        '-b',"--basis", type=str, choices=['x', 'y', 'z'], default='z',
        help="Tomography basis for Bob's qubit"
    )
    args = parser.parse_args()
    print(vars(args))
 
    # Build circuit
    qcTele = circTeleport_noFF(args.basis, args.secretState)

    # Run on Aer simulator
    backend = AerSimulator()
    sampler = Sampler(mode=backend)
    job = sampler.run([qcTele], shots=args.shots)
    result = job.result()
    counts_raw = result[0].data.c.get_counts()

    print(qcTele.draw(output="text", idle_wires=False))
    print("Raw counts (c2 c1 c0):", counts_raw)

    # Post-process correction
    counts_corrected = post_process_counts(counts_raw, meas_basis=args.basis)
    print("Corrected Bob counts:", counts_corrected)

    # Evaluate
    evaluate_results(counts_corrected, args.secretState, args.basis, nSigThr=3)
