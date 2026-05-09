#!/usr/bin/env python3
"""Ideal Bell-state shot sampler in SQUIN.

Builds a 2-qubit Bell circuit, optionally prints the Cirq/SQUIN circuit, runs
PyQrack DynamicMemorySimulator one shot at a time, and prints measured counts.
"""

import argparse
from typing import Any

from bloqade.types import MeasurementResult
from bloqade import squin
from kirin.dialects import ilist
from Util_Squin import print_counts, print_kernel_circuit, run_kernel_shots


@squin.kernel
def main() -> ilist.IList[MeasurementResult, Any]:
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return squin.broadcast.measure(q)


def print_circuit(verb: int) -> None:
    print_kernel_circuit(main, (), verb)


def run() -> None:
    parser = argparse.ArgumentParser()
    prs = parser.add_argument
    prs("--shots", type=int, default=1000)
    prs("-v", "--verb", type=int, default=1, help="increase output verbosity")
    args = parser.parse_args()

    print_circuit(args.verb)

    print("Simulator: PyQrack DynamicMemorySimulator (state vector, ideal)")
    counts = run_kernel_shots(main, (), args.shots)
    print(f"Ideal Bell-state shot results ({args.shots} shots)")
    print_counts(counts)


if __name__ == "__main__":
    run()
