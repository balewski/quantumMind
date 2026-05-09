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


@squin.kernel
def main(n: int) -> ilist.IList[MeasurementResult, Any]:
    q = squin.qalloc(n)
    squin.h(q[0])
    for i in range(n - 1):
        squin.cx(q[i], q[i + 1])
    return squin.broadcast.measure(q)


def print_circuit(verb: int, num_qubits: int) -> None:
    print_kernel_circuit(main, (num_qubits,), verb)


def run() -> None:
    parser = argparse.ArgumentParser()
    prs = parser.add_argument
    prs("--shots", type=int, default=1000)
    prs("-q", "--qubits", type=int, default=2, help="number of qubits")
    prs("-v", "--verb", type=int, default=1, help="increase output verbosity")
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    if args.qubits < 1:
        raise ValueError("number of qubits must be at least 1")

    print_circuit(args.verb, args.qubits)

    counts = run_kernel_shots(main, (args.qubits,), args.shots)
    print(f"Ideal {args.qubits}-qubit GHZ shot results ({args.shots} shots)")
    print_counts(counts)


if __name__ == "__main__":
    run()
