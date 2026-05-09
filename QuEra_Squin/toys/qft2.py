#!/usr/bin/env python3
"""Noisy inverse-QFT benchmark in SQUIN.

Adds the ghz2.py noise model to qft1.py: 1Q/2Q depolarization, atom erasure
after 2Q gate checkpoints, and readout bit flips. Fidelity uses no-erasure shots.
"""

import argparse
from typing import Any

import numpy as np
from bloqade import squin
from bloqade.types import MeasurementResult, Qubit
from kirin.dialects import ilist
from qft1 import post_process_counts, print_phase_table
from Util_Squin import (
    get_lost_measurement_result,
    print_counts,
    print_kernel_circuit,
    print_loss_summary,
    run_kernel_shots,
    split_erasure_counts,
    validate_probability,
)


def generate_qft_noise_program(default_num_qubits: int = 3):
    @squin.kernel
    def apply_noisy_shift(angle: float, qubit: Qubit, noise_1q: float) -> None:
        squin.shift(angle, qubit)
        squin.depolarize(p=noise_1q, qubit=qubit)

    @squin.kernel
    def apply_noisy_h(qubit: Qubit, noise_1q: float) -> None:
        squin.h(qubit)
        squin.depolarize(p=noise_1q, qubit=qubit)

    @squin.kernel
    def apply_noisy_cx(
        control: Qubit,
        target: Qubit,
        all_qubits: ilist.IList[Qubit, Any],
        noise_2q: float,
        noise_erasure: float,
    ) -> None:
        squin.cx(control, target)
        squin.depolarize2(noise_2q, control, target)
        # Same erasure model as ghz2.py: one independent loss trial per qubit
        # after each 2Q gate checkpoint.
        squin.broadcast.qubit_loss(noise_erasure, all_qubits)

    @squin.kernel
    def apply_noisy_cphase(
        angle: float,
        control: Qubit,
        target: Qubit,
        all_qubits: ilist.IList[Qubit, Any],
        noise_1q: float,
        noise_2q: float,
        noise_erasure: float,
    ) -> None:
        apply_noisy_shift(angle / 2.0, control, noise_1q)
        apply_noisy_cx(control, target, all_qubits, noise_2q, noise_erasure)
        apply_noisy_shift(-angle / 2.0, target, noise_1q)
        apply_noisy_cx(control, target, all_qubits, noise_2q, noise_erasure)
        apply_noisy_shift(angle / 2.0, target, noise_1q)

    @squin.kernel
    def apply_noisy_inverse_qft(
        q: ilist.IList[Qubit, Any],
        n: int,
        noise_1q: float,
        noise_2q: float,
        noise_erasure: float,
    ) -> None:
        for target_offset in range(n):
            target = n - 1 - target_offset
            for control in range(target + 1, n):
                angle = -np.pi / (2 ** (control - target))
                apply_noisy_cphase(
                    angle, q[control], q[target], q, noise_1q, noise_2q, noise_erasure
                )
            apply_noisy_h(q[target], noise_1q)

    @squin.kernel
    def qft_noise_prog(
        n: int,
        freq: int,
        noise_1q: float,
        noise_2q: float,
        noise_erasure: float,
        noise_readout: float,
    ) -> ilist.IList[MeasurementResult, Any]:
        q = squin.qalloc(n)

        # Prepare QFT(|freq>) as a product state, with 1Q noise after each prep gate.
        for j in range(n):
            angle = 2 * np.pi * freq / (2 ** (n - j))
            apply_noisy_h(q[j], noise_1q)
            apply_noisy_shift(angle, q[j], noise_1q)

        apply_noisy_inverse_qft(q, n, noise_1q, noise_2q, noise_erasure)
        squin.broadcast.bit_flip(noise_readout, q)
        return squin.broadcast.measure(q)

    return qft_noise_prog, default_num_qubits


def main() -> None:
    qft_noise_prog, default_num_qubits = generate_qft_noise_program()

    parser = argparse.ArgumentParser(description="Noisy SQUIN inverse-QFT benchmark")
    prs = parser.add_argument
    prs("-q", "--qubits", type=int, default=default_num_qubits, help="number of qubits")
    prs("-k", "--freq", type=int, default=1, help="input frequency to decode")
    prs("-n", "--shots", type=int, default=2000, help="number of shots")
    prs("-v", "--verb", type=int, default=1, help="increase output verbosity")
    prs("--noise_1q", type=float, default=0.0, help="1-qubit depolarizing probability")
    prs("--noise_2q", type=float, default=0.0, help="2-qubit depolarizing probability")
    prs("--noise_erasure", type=float, default=0.0, help="independent per-qubit loss probability after each 2-qubit gate")
    prs("--noise_readout", type=float, default=0.05, help="readout bit-flip probability")
    args = parser.parse_args()

    if args.qubits < 1:
        raise ValueError("number of qubits must be at least 1")
    if args.freq < 0 or args.freq >= 2 ** args.qubits:
        raise ValueError("freq must satisfy 0 <= freq < 2**qubits")
    validate_probability("--noise_1q", args.noise_1q)
    validate_probability("--noise_2q", args.noise_2q)
    validate_probability("--noise_erasure", args.noise_erasure)
    validate_probability("--noise_readout", args.noise_readout)

    kernel_args = (
        args.qubits,
        args.freq,
        args.noise_1q,
        args.noise_2q,
        args.noise_erasure,
        args.noise_readout,
    )
    print_phase_table(args.qubits, args.freq)
    print_kernel_circuit(qft_noise_prog, kernel_args, args.verb)

    print("Simulator: PyQrack DynamicMemorySimulator (state vector, stochastic noise)")
    lost_result = get_lost_measurement_result()
    counts = run_kernel_shots(
        qft_noise_prog, kernel_args, args.shots, loss_m_result=lost_result
    )

    print(f"\n--- Noisy inverse-QFT benchmark (Size={args.qubits}) ---")
    print(f"Input Frequency: {args.freq}")
    print(
        f"noise: 1q={args.noise_1q:g}, 2q={args.noise_2q:g}, "
        f"erasure={args.noise_erasure:g}, readout={args.noise_readout:g}"
    )
    if args.noise_erasure > 0:
        if lost_result is None:
            print("erasure readout: this PyQrack install does not expose a Lost result")
        else:
            print("erasure readout: lost atoms are reported as L")

    counts_no_erasure, counts_with_erasure = split_erasure_counts(counts)
    shots_no_erasure = sum(counts_no_erasure.values())
    shots_with_erasure = sum(counts_with_erasure.values())
    print_loss_summary(counts)
    print(f"shots requested: {args.shots}")
    print(f"shots without erasure: {shots_no_erasure}")
    print(f"shots with erasure: {shots_with_erasure}")
    print(f"\nCounts without erasure {shots_no_erasure}:")
    print_counts(counts_no_erasure)
    print(f"\nCounts with erasure {shots_with_erasure}:")
    print_counts(counts_with_erasure)
    args_no_erasure = argparse.Namespace(**vars(args))
    args_no_erasure.shots = shots_no_erasure
    post_process_counts(counts_no_erasure, args_no_erasure)


if __name__ == "__main__":
    main()
