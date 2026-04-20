#!/usr/bin/env python3
"""Measure repeated 1-qubit gate action on FakeTorino."""

import argparse
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated 1-qubit X/H gates on FakeTorino."
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=400_000,
        help="Number of shots per circuit.",
    )
    parser.add_argument(
        "--gate",
        type=str,
        choices=["x", "h"],
        default="x",
        help="Single-qubit gate to repeat.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=16,
        help="Maximum gate power. Must be even.",
    )
    args = parser.parse_args()
    if args.repeat < 2 or args.repeat % 2 != 0:
        raise ValueError(f"--repeat must be an even integer >= 2, got {args.repeat}")
    return args


def prep_circuits(gate_name, repeat_max):
    """Build circuits with the chosen gate repeated 0,2,...,repeat_max times."""
    circuits = []
    repeat_values = list(range(0, repeat_max + 1, 2))

    for repeat_count in repeat_values:
        qc = QuantumCircuit(1, 1, name=f"{gate_name}^{repeat_count}")
        qc.x(0)  # Start from |1>.
        qc.barrier()
        for _ in range(repeat_count):
            getattr(qc, gate_name)(0)
            qc.barrier()
        qc.measure(0, 0)
        circuits.append(qc)

    return circuits, repeat_values


def measure(circuits, num_shots):
    """Transpile and execute all circuits on a FakeTorino-matched simulator."""
    backend = FakeTorino()
    simulator = AerSimulator.from_backend(backend)
    transpiled = transpile(
        circuits,
        backend=backend,
        optimization_level=1,
        seed_transpiler=123,
    )

    start_time = time()
    job = simulator.run(transpiled, shots=num_shots)
    result = job.result()
    elapsed = time() - start_time

    counts_list = []
    for idx in range(len(circuits)):
        counts_list.append(result.get_counts(idx))

    print(f"Simulation elapsed seconds: {elapsed:.3f}")

    return {
        "backend_name": backend.name,
        "transpiled_circuits": transpiled,
        "counts_list": counts_list,
        "num_shots": num_shots,
        "elapsed_seconds": elapsed,
    }


def postprocessing(measurement_data, gate_name, repeat_values):
    """Compute P(1), fit a straight line, and save the plot."""
    prob_one = []
    prob_err = []

    for repeat_count, counts in zip(repeat_values, measurement_data["counts_list"]):
        ones = counts.get("1", 0)
        shots = sum(counts.values())
        p1 = ones / shots if shots else 0.0
        err = np.sqrt(p1 * (1.0 - p1) / shots) if shots else 0.0
        prob_one.append(p1)
        prob_err.append(err)
        print(f"repeat={repeat_count:2d} counts={counts} p(1)={p1:.4f} sigma={err:.4f}")

    x_vals = np.asarray(repeat_values, dtype=float)
    y_vals = np.asarray(prob_one, dtype=float)
    if len(x_vals) >= 2:
        fit_slope, fit_intercept = np.polyfit(x_vals, y_vals, deg=1)
    else:
        fit_slope, fit_intercept = 0.0, float(y_vals[0])
    fit_line = fit_slope * x_vals + fit_intercept

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gate1Q_power_{gate_name}.png"

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(
        x_vals,
        y_vals,
        yerr=prob_err,
        fmt="o",
        capsize=4,
        label="Measured P(1)",
    )
    plt.plot(
        x_vals,
        fit_line,
        "-",
        label=f"Linear fit: p={fit_slope:.4g}*n + {fit_intercept:.4g}",
    )
    plt.xlabel("Number of gate applications")
    plt.ylabel("Probability of measuring 1")
    plt.title(f"{gate_name.upper()} gate power on FakeTorino")
    #plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

    print(f"\nBackend: {measurement_data['backend_name']}")
    print(f"Linear fit slope={fit_slope:.6f}, intercept={fit_intercept:.6f}")
    print(f"Saved plot: {out_path}")

    return {
        "prob_one": y_vals,
        "prob_err": np.asarray(prob_err),
        "fit_slope": fit_slope,
        "fit_intercept": fit_intercept,
        "plot_path": out_path,
    }


def main():
    args = parse_args()
    for arg in vars(args):
        print('myArg:', arg, getattr(args, arg))

    circuits, repeat_values = prep_circuits(args.gate, args.repeat)
    print(f"Prepared {len(circuits)} circuits for gate={args.gate} repeats={repeat_values}")
    for circuit in circuits:
        print(circuit.draw("text"))

    measurement_data = measure(circuits, args.shots)
    postprocessing(measurement_data, args.gate, repeat_values)


if __name__ == "__main__":
    main()
