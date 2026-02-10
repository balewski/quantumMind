#!/usr/bin/env python3
"""
This program benchmarks qubit SPAM by running one "Reset, X-gate, Measure" round per qubit.
It constructs a Qiskit circuit where every qubit independently undergoes this sequence, recording
results in separate classical registers. The script executes this circuit on either an ideal
simulator or real IBM hardware. Post-processing computes Prob(0) and its binomial standard error
from measurement data, and generates a scatter plot of calibration prep1meas0 vs measured Prob(0).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler

def parse_args():
    """Parse command-line arguments for the X-Measure-Reset benchmark."""
    parser = argparse.ArgumentParser(description="X-Measure-Reset Sequence Benchmark")
    parser.add_argument('--size', type=int, default=5, help="Number of qubits")
    parser.add_argument('-n', "--shots", type=int, default=40_000, help="Number of simulation shots")
    parser.add_argument('-c', "--printCirc", action="store_true", help="Print circuit")
    parser.add_argument('-b', '--backend', default='ideal', help="quantum backend")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Seed for random generator")
    parser.add_argument('--pct', type=float, default=30.0,
                        help="Percentile for median error bars (e.g., 5 -> 5th/95th)")
    return parser.parse_args()

def build_x_meas_reset_circuit(n_qubits):
    """Construct a circuit performing one X, measure, reset cycle per qubit.

    Returns a `QuantumCircuit` with one classical register per qubit (`c0`, `c1`, ...).
    """
    print(f"Building circuit: {n_qubits} qubits, 1 round (Reset-X-Measure)")
    qr = QuantumRegister(n_qubits, 'q')
    crs = [ClassicalRegister(1, name=f"c{i}") for i in range(n_qubits)]
    qc = QuantumCircuit(qr, *crs)

    for q in range(n_qubits):
        qc.x(qr[q])
        qc.measure(qr[q], crs[q][0])
    return qc

def get_prep1meas0_errors(backend, phys_qubits):
    """Return {physQ: p10} where p10 = P(meas 0 | prep 1) from backend calibration."""
    p10_map = {}
    if backend is None:
        return p10_map
    if getattr(backend, "name", None) == "aer_simulator":
        return p10_map
    props = backend.properties()
    if not props:
        return p10_map

    for q in sorted(set(phys_qubits)):
        try:
            p10 = props.qubit_property(q, 'prob_meas0_prep1')[0]
            p10_map[q] = p10
        except Exception:
            p10_map[q] = None
    return p10_map


def compute_spam_stats(backend, result_data, n_qubits, phys_layout):
    """Compute per-qubit stats for the SPAM table and plot."""
    prep1meas0 = get_prep1meas0_errors(backend, phys_layout[:n_qubits])
    rows = []
    for i in range(n_qubits):
        creg = getattr(result_data, f"c{i}")
        counts = creg.get_counts()
        total_samples = sum(counts.values())
        count_0 = int(counts.get("0", 0))
        prob_0 = count_0 / total_samples if total_samples > 0 else 0.0

        if 0 < prob_0 < 1 and total_samples > 0:
            err_0 = float(np.sqrt(prob_0 * (1 - prob_0) / total_samples))
        else:
            err_0 = 1.0 / total_samples if total_samples > 0 else 0.0

        phys_id = phys_layout[i]
        aerr = prep1meas0.get(phys_id, None)
        rows.append({
            "logQ": i,
            "physQ": phys_id,
            "prep1meas0": aerr,
            "prob0": prob_0,
            "std": err_0,
        })
    return rows


def print_spam_table(rows):
    title = "SPAM summary: calibration prep1meas0 vs measured Prob(0)"
    bar_eq = "=" * 72
    bar_dash = "-" * 72

    def fmt4(x):
        try:
            if x is None or not np.isfinite(float(x)):
                return "   nan  "
            return f"{float(x):0.4f}"
        except Exception:
            return "   nan  "

    print("\n" + bar_eq)
    print(title)
    print(bar_eq)
    print("LogQ   | PhysQ  | prep1m0  | Prob(0)    | std prob  ")
    print(bar_dash)
    for r in rows:
        logq = int(r["logQ"])
        physq = int(r["physQ"])
        prep = fmt4(r.get("prep1meas0", None))
        prob0 = fmt4(r.get("prob0", None))
        std = fmt4(r.get("std", None))
        print(f"{logq:<6} | {physq:<6} | {prep:<8} | {prob0:<10} | {std:<9}")


def plot_spam(rows, pct=10.0, out_png=None, run_date=None, qpu_name=None):
    """Scatter plot AvgErr vs Prob(0) with error bars, median marker, and fit line."""
    # Keep only rows with calibration available
    xs = np.array([r["prep1meas0"] for r in rows], dtype=float)
    ys = np.array([r["prob0"] for r in rows], dtype=float)
    yerrs = np.array([r["std"] for r in rows], dtype=float)
    phys_ids = np.array([r["physQ"] for r in rows], dtype=int)

    good = np.isfinite(xs)
    xs = xs[good]
    ys = ys[good]
    yerrs = yerrs[good]
    phys_ids = phys_ids[good]

    if xs.size == 0:
        print("\nNo calibration prep1meas0 available on this backend; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, ys, yerr=yerrs, fmt='o', ms=5, capsize=3, alpha=0.8, label='Qubits')

    # dashed x=y (fixed extent)
    ax.plot([0.0, 0.1], [0.0, 0.1], '--', color='gray', lw=1, label='x=y')

    # median marker with percentile error bars
    p = float(pct)
    p_lo, p_hi = p, 100.0 - p
    med_x = float(np.median(xs))
    med_y = float(np.median(ys))
    x_lo, x_hi = np.percentile(xs, [p_lo, p_hi])
    y_lo, y_hi = np.percentile(ys, [p_lo, p_hi])
    ax.errorbar([med_x], [med_y],
                xerr=[[med_x - x_lo], [x_hi - med_x]],
                yerr=[[med_y - y_lo], [y_hi - med_y]],
                fmt='s', ms=10, mfc='none', mec='black', mew=2,
                ecolor='black', elinewidth=2, capsize=5,
                label=f'median ± [{p_lo:.0f},{p_hi:.0f}]%')

    # (dropped) correlation/fit line

    ax.set_xlabel("prep1meas0 (calibration)")
    ax.set_ylabel("Prob(0) (measured) ± std")
    title_parts = ["SPAM: calibration vs measurement"]
    if run_date:
        title_parts.append(str(run_date))
    if qpu_name:
        title_parts.append(str(qpu_name))
    ax.set_title(" | ".join(title_parts))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Physical qubit list across top of plot (few rows)
    per_row = 12
    q_list = [str(q) for q in phys_ids.tolist()]
    lines = []
    for k in range(0, len(q_list), per_row):
        lines.append(" ".join(q_list[k:k + per_row]))
    txt = "PhysQ: " + "\n      ".join(lines[:4])  # cap at a few rows
    ax.text(0.5, 0.98, txt, transform=ax.transAxes,
            ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

    # axis limits and aspect ratio
    lo = -0.001
    if qpu_name == "ibm_miami":
        hi = 0.035
    elif qpu_name == "ibm_marrakesh":
        hi = 0.025
    elif qpu_name == "ibm_pittsburgh":
        hi = 0.01
    else:
        hi = 0.01
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if out_png:
        out_dir = os.path.dirname(out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_png, dpi=200)
        print(f"\nSaved plot: {out_png}")
    plt.show()

if __name__ == "__main__":
    args = parse_args()

    # 1. Build
    qc = build_x_meas_reset_circuit(args.size)
    if args.printCirc: print(qc)
    
    # 2. Backend Setup
    if args.backend == 'ideal':
        print(f"\nRunning simulation (Ideal Aer) with {args.shots} shots...")
        backend = AerSimulator(seed_simulator=args.seed)
        qc_run = qc 
        phys_layout = list(range(args.size))
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit import transpile
        print(f'\nUsing real HW backend: {args.backend} ...')
        service = QiskitRuntimeService()
        backend = service.backend(args.backend)
        qc_run = transpile(qc, backend)
        phys_layout = qc_run.layout.final_index_layout(filter_ancillas=True)
        print(f"Transpilation complete. Physical Layout: {phys_layout}")
        
    # 3. Execution
    print(f'\nRunning sampler on {backend.name}  nq={args.size} ...')
    sampler = Sampler(mode=backend)
    job = sampler.run([qc_run], shots=args.shots)
    result = job.result()
    print('Job done. shots:',args.shots)
    
    # 4. Analysis table
    rows = compute_spam_stats(backend, result[0].data, args.size, phys_layout)
    print_spam_table(rows)
    back_name = backend.name if isinstance(backend.name, str) else backend.name()
    run_date = datetime.now().strftime("%Y-%m-%d")
    out_png = f"out/spam_{run_date}_{back_name}_nq{args.size}.png"
    plot_spam(rows, pct=args.pct, out_png=out_png, run_date=run_date, qpu_name=back_name)
