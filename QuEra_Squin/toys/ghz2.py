#!/usr/bin/env python3
"""Noisy n-qubit GHZ shot sampler in SQUIN.

Extends ghz1.py with 1Q/2Q depolarization, atom erasure after each 2Q gate,
and readout bit-flip noise. Counts are split by detected erasure markers.
"""

import argparse
from typing import Any

from bloqade import squin
from bloqade.types import MeasurementResult
from kirin.dialects import ilist
from Util_Squin import (
    get_lost_measurement_result,
    print_counts,
    print_kernel_circuit,
    print_loss_summary,
    run_kernel_shots,
    split_erasure_counts,
    validate_probability,
)


def generate_ghz_noise_program(default_num_qubits: int = 2):
    @squin.kernel
    def ghz_noise_prog(
        n: int,
        noise_1q: float,
        noise_2q: float,
        noise_erasure: float,
        noise_readout: float,
    ) -> ilist.IList[MeasurementResult, Any]:
        q = squin.qalloc(n)

        # Prepare the GHZ branch point, then apply optional 1Q depolarizing noise.
        squin.h(q[0])
        squin.depolarize(p=noise_1q, qubit=q[0])

        # Extend the GHZ chain; each CX can be followed by optional 2Q depolarizing noise.
        for i in range(n - 1):
            squin.cx(q[i], q[i + 1])
            squin.depolarize2(noise_2q, q[i], q[i + 1])
            # One independent erasure/loss trial per qubit after each 2Q gate.
            # With many 2Q gates, total loss probability compounds across checkpoints.
            squin.broadcast.qubit_loss(noise_erasure, q)

        # Model readout error as a classical-looking bit flip just before measurement.
        squin.broadcast.bit_flip(noise_readout, q)
        return squin.broadcast.measure(q)

    return ghz_noise_prog, default_num_qubits


def main() -> None:
    ghz_noise_prog, default_num_qubits = generate_ghz_noise_program()

    parser = argparse.ArgumentParser()
    prs = parser.add_argument
    prs("--shots", type=int, default=1000)
    prs("-q", "--qubits", type=int, default=default_num_qubits, help="number of qubits")
    prs("-v", "--verb", type=int, default=1, help="increase output verbosity")
    prs("--noise_1q", type=float, default=0.0, help="1-qubit depolarizing probability")
    prs("--noise_2q", type=float, default=0.0, help="2-qubit depolarizing probability")
    prs("--noise_erasure", type=float, default=0.5, help="independent per-qubit loss probability after each 2-qubit gate")
    prs("--noise_readout", type=float, default=0.0, help="readout bit-flip probability")
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    if args.qubits < 1:
        raise ValueError("number of qubits must be at least 1")
    validate_probability("--noise_1q", args.noise_1q)
    validate_probability("--noise_2q", args.noise_2q)
    validate_probability("--noise_erasure", args.noise_erasure)
    validate_probability("--noise_readout", args.noise_readout)
   
    kernel_args = (
        args.qubits,
        args.noise_1q,
        args.noise_2q,
        args.noise_erasure,
        args.noise_readout,
    )
    print_kernel_circuit(ghz_noise_prog, kernel_args, args.verb)

    # run_kernel_shots uses PyQrack DynamicMemorySimulator: state-vector shots,
    # not a density-matrix simulator. Each task.run() produces one noisy shot.
    # This installed PyQrack path reports lost atoms through its loss_m_result
    # policy; in this container that defaults to ordinary "1" measurements.
    print("Simulator: PyQrack DynamicMemorySimulator (state vector, stochastic noise)")
    lost_result = get_lost_measurement_result()
    counts = run_kernel_shots(
        ghz_noise_prog, kernel_args, args.shots, loss_m_result=lost_result
    )
    print(f"Noisy {args.qubits}-qubit GHZ shot results ({args.shots} shots)")
    print(
        f"noise: 1q={args.noise_1q:g}, 2q={args.noise_2q:g}, "
        f"erasure={args.noise_erasure:g}, "
        f"readout={args.noise_readout:g}"
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


if __name__ == "__main__":
    main()
