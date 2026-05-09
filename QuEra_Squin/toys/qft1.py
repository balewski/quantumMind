#!/usr/bin/env python3
"""Ideal inverse-QFT benchmark in SQUIN.

Prepares the Fourier product state for integer k, applies an inverse QFT,
samples the ideal state-vector simulator, and reports decoding fidelity.
"""

import argparse
from typing import Any

import numpy as np
from bloqade import squin
from bloqade.types import MeasurementResult, Qubit
from kirin.dialects import ilist
from Util_Squin import print_kernel_circuit, run_kernel_shots


def generate_qft_program(default_num_qubits: int = 3):
    @squin.kernel
    def apply_cphase(angle: float, control: Qubit, target: Qubit) -> None:
        squin.shift(angle / 2.0, control)
        squin.cx(control, target)
        squin.shift(-angle / 2.0, target)
        squin.cx(control, target)
        squin.shift(angle / 2.0, target)

    @squin.kernel
    def apply_inverse_qft(q: ilist.IList[Qubit, Any], n: int) -> None:
        for target_offset in range(n):
            target = n - 1 - target_offset
            for control in range(target + 1, n):
                angle = -np.pi / (2 ** (control - target))
                apply_cphase(angle, q[control], q[target])
            squin.h(q[target])

    @squin.kernel
    def qft_prog(n: int, freq: int) -> ilist.IList[MeasurementResult, Any]:
        q = squin.qalloc(n)

        # Prepare QFT(|freq>) as a product state, avoiding a costly forward QFT.
        for j in range(n):
            angle = 2 * np.pi * freq / (2 ** (n - j))
            squin.h(q[j])
            squin.shift(angle, q[j])

        apply_inverse_qft(q, n)
        return squin.broadcast.measure(q)

    return qft_prog, default_num_qubits


def print_phase_table(num_qubits: int, freq: int) -> None:
    print(f"Preparing Fourier state |~{freq}> on {num_qubits} qubits:")
    for j in range(num_qubits):
        angle = 2 * np.pi * freq / (2 ** (num_qubits - j))
        print(f"  Qubit {j}: phi={angle:.4f} rad")


def post_process_counts(counts: dict[str, int], args: argparse.Namespace) -> None:
    print("\n--- Post-processing counts ---")
    target_str = f"{args.freq:0{args.qubits}b}"
    success_count = counts.get(target_str, 0)
    fidelity = success_count / args.shots if args.shots else 0.0
    infidelity = 1.0 - fidelity

    print(f"Target State: |{target_str}>")
    print(f"Success Count: {success_count} / {args.shots}")
    print(f"Fidelity:   {fidelity:.4f}")
    print(f"Infidelity: {infidelity:.4f}")
    print("\nTop measured states:")
    for state, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]:
        mark = " (TARGET)" if state == target_str else ""
        print(f"  |{state}> : {count}{mark}")


def main() -> None:
    qft_prog, default_num_qubits = generate_qft_program()

    parser = argparse.ArgumentParser(description="SQUIN inverse-QFT fidelity benchmark")
    prs = parser.add_argument
    prs("-q", "--qubits", type=int, default=default_num_qubits, help="number of qubits")
    prs("-k", "--freq", type=int, default=1, help="input frequency to decode")
    prs("-n", "--shots", type=int, default=2000, help="number of shots")
    prs("-v", "--verb", type=int, default=1, help="increase output verbosity")
    args = parser.parse_args()

    if args.qubits < 1:
        raise ValueError("number of qubits must be at least 1")
    if args.freq < 0 or args.freq >= 2 ** args.qubits:
        raise ValueError("freq must satisfy 0 <= freq < 2**qubits")

    kernel_args = (args.qubits, args.freq)
    print_phase_table(args.qubits, args.freq)
    print_kernel_circuit(qft_prog, kernel_args, args.verb)

    print("Simulator: PyQrack DynamicMemorySimulator (state vector, ideal)")
    counts = run_kernel_shots(qft_prog, kernel_args, args.shots)

    print(f"\n--- Benchmarking inverse QFT (Size={args.qubits}) ---")
    print(f"Input Frequency: {args.freq}")
    print("counts:", dict(sorted(counts.items())))
    post_process_counts(counts, args)


if __name__ == "__main__":
    main()
