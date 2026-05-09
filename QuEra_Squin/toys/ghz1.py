#!/usr/bin/env python3
"""Ideal n-qubit GHZ shot sampler in SQUIN.

Builds an H/CX-chain GHZ circuit with configurable qubit count, optionally
prints the Cirq/SQUIN circuit, runs ideal state-vector shots, and prints counts.
"""

import argparse
from typing import Any

from bloqade import squin
from bloqade.types import MeasurementResult
from kirin.dialects import ilist
from Util_Squin import print_counts, print_kernel_circuit, run_kernel_shots


def generate_ghz_program(default_num_qubits: int = 2):
    @squin.kernel
    def ghz_prog(n: int) -> ilist.IList[MeasurementResult, Any]:
        q = squin.qalloc(n)
        squin.h(q[0])
        for i in range(n - 1):
            squin.cx(q[i], q[i + 1])
        return squin.broadcast.measure(q)

    return ghz_prog, default_num_qubits


def print_circuit(ghz_prog: Any, verb: int, num_qubits: int) -> None:
    print_kernel_circuit(ghz_prog, (num_qubits,), verb)


def main() -> None:
    ghz_prog, default_num_qubits = generate_ghz_program()

    parser = argparse.ArgumentParser()
    prs = parser.add_argument
    prs("--shots", type=int, default=1000)
    prs("-q", "--qubits", type=int, default=default_num_qubits, help="number of qubits")
    prs("-v", "--verb", type=int, default=1, help="increase output verbosity")
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    if args.qubits < 1:
        raise ValueError("number of qubits must be at least 1")

    print_circuit(ghz_prog, args.verb, args.qubits)

    counts = run_kernel_shots(ghz_prog, (args.qubits,), args.shots)
    print(f"Ideal {args.qubits}-qubit GHZ shot results ({args.shots} shots)")
    print_counts(counts)


if __name__ == "__main__":
    main()
