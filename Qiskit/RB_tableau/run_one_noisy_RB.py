#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Clifford
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, reset_error
from toolbox.Util_NumpyIOv1 import read_data_npz, write_data_npz

logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
logging.getLogger("qiskit.compiler").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simplified local noisy RB runner for canned RB sequences."
    )
    parser.add_argument("--basePath", default="out", help="head dir for set of experiments")
    parser.add_argument("--rbType", choices=["std1q", "std2q", "int2q", "std3q", "int3q"], default="std1q")
    parser.add_argument("--inputRB", required=True, help="RB input stem or .npz path")
    parser.add_argument("--verb", type=int, default=1, help="verbosity level")
    parser.add_argument("--expName", default=None, help="name used for exported outputs")
    parser.add_argument("--noBarrier", action="store_true", default=False)
    parser.add_argument("--totShot", type=int, default=2000, help="shots per RB circuit")
    parser.add_argument(
        "--exportQPY",
        action="store_true",
        default=False,
        help="export ideal and transpiled circuits as QPY",
    )
    parser.add_argument(
        "--printCirc",
        type=str,
        default="",
        help="print first circuit: i=ideal, d=decomposed ideal, t=transpiled",
    )
    parser.add_argument(
        "--noise_errIdle",
        type=float,
        default=0.0,
        help="Idle gate depolarizing error",
    )
    parser.add_argument(
        "--noise_err1Q",
        type=float,
        default=1e-4,
        help="Other 1-qubit gate depolarizing error",
    )
    parser.add_argument(
        "--noise_err2Q",
        type=float,
        default=4e-4,
        help="2-qubit (CX) gate depolarizing error",
    )
    parser.add_argument(
        "--noise_errRead",
        type=float,
        default=2e-4,
        help="Readout error probability",
    )
    parser.add_argument(
        "--noise_errReset",
        type=float,
        default=None,
        help="Reset error probability (defaults to errRead if None)",
    )
    return parser.parse_args()


def rb_num_qubits(rb_type):
    if rb_type == "std1q":
        return 1
    if rb_type in ("std2q", "int2q"):
        return 2
    if rb_type in ("std3q", "int3q"):
        return 3
    raise ValueError(f"Unsupported rbType: {rb_type}")


def rb_subdir(rb_type):
    if "1q" in rb_type:
        return "inputRB_1Q"
    if "3q" in rb_type:
        return "inputRB_3Q"
    return "inputRB_2Q"


def resolve_input_file(input_rb, rb_type):
    direct = Path(input_rb)
    if direct.exists():
        return direct.resolve()

    if direct.suffix != ".npz":
        with_suffix = Path(f"{input_rb}.{rb_type}.npz")
        if with_suffix.exists():
            return with_suffix.resolve()

    fname = f"{input_rb}.{rb_type}.npz" if direct.suffix != ".npz" else direct.name
    candidates = [
        Path.cwd() / fname,
        Path.cwd() / rb_subdir(rb_type) / fname,
        Path.cwd() / "out" / rb_subdir(rb_type) / fname,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(f"RB file not found for inputRB={input_rb}, rbType={rb_type}")


def load_rb_tableaus(args):
    direct = Path(args.inputRB)
    if direct.exists():
        inp_path = direct.resolve()
    else:
        inp_path = Path(args.basePath) / rb_subdir(args.rbType) / f"{args.inputRB}.{args.rbType}.npz"
        if not inp_path.exists():
            inp_path = resolve_input_file(args.inputRB, args.rbType)
    rb_data, rb_meta = read_data_npz(str(inp_path), verb=max(0, args.verb - 1))
    if "rb_tableaus" not in rb_data:
        raise KeyError(f"Missing 'rb_tableaus' in {inp_path}")
    rb_tableaus = rb_data["rb_tableaus"]
    if rb_tableaus.ndim != 4:
        raise ValueError(f"Unexpected rb_tableaus shape: {rb_tableaus.shape}")
    return inp_path, rb_tableaus, rb_meta


def build_one_rb_circuit(tableaus, n_qubits, no_barrier, interleaved):
    qc = QuantumCircuit(n_qubits, n_qubits)
    for idx, tab in enumerate(tableaus):
        if not no_barrier:
            if interleaved:
                if idx % 2 == 0:
                    qc.barrier(label=f"u{idx // 2}")
                else:
                    qc.barrier()
            else:
                qc.barrier(label=f"u{idx}")
        qc.compose(Clifford(tab).to_circuit(), qubits=range(n_qubits), inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def build_rb_circuits(rb_tableaus, rb_type, no_barrier):
    n_qubits = rb_num_qubits(rb_type)
    interleaved = rb_type in ("int2q", "int3q")
    qc_list = []
    for seq in rb_tableaus:
        qc_list.append(build_one_rb_circuit(seq, n_qubits, no_barrier, interleaved))
    return qc_list, n_qubits


def build_noise_model(args):
    err_idle = args.noise_errIdle
    err_1q = args.noise_err1Q
    err_2q = args.noise_err2Q
    err_read = args.noise_errRead
    err_reset = args.noise_errReset if args.noise_errReset is not None else err_read

    noise_model = NoiseModel()
    if err_idle > 0:
        noise_model.add_all_qubit_quantum_error(depolarizing_error(err_idle, 1), ["id"])
    if err_1q > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(err_1q, 1),
            ["u", "u1", "u2", "u3", "x", "h", "rx", "ry", "rz", "sx", "z", "s", "sdg"],
        )
    if err_2q > 0:
        noise_model.add_all_qubit_quantum_error(depolarizing_error(err_2q, 2), ["cx", "cz"])
    if err_read > 0:
        readout = ReadoutError([[1 - err_read, err_read], [err_read, 1 - err_read]])
        noise_model.add_all_qubit_readout_error(readout)
    if err_reset > 0:
        noise_model.add_all_qubit_quantum_error(reset_error(1 - err_reset, err_reset), ["reset"])

    noise_conf = {
        "noise_errIdle": err_idle,
        "noise_err1Q": err_1q,
        "noise_err2Q": err_2q,
        "noise_errRead": err_read,
        "noise_errReset": err_reset,
    }
    return noise_model, noise_conf


def transpile_circuits(qc_list, n_qubits, noise_model):
    basis_gates = ["id", "rz", "sx", "x"]
    if n_qubits > 1:
        basis_gates.append("cx")
    backend = AerSimulator(noise_model=noise_model, seed_simulator=42)
    qc_t = transpile(
        qc_list,
        backend=backend,
        basis_gates=basis_gates,
        optimization_level=3,
        seed_transpiler=42,
    )
    return backend, qc_t


def print_requested_circuits(args, qc_ideal, qc_transpiled):
    if "i" in args.printCirc:
        print(qc_ideal.draw())
    if "d" in args.printCirc:
        print("decomposed ideal circuit:")
        print(qc_ideal.decompose())
    if "t" in args.printCirc:
        print(qc_transpiled.draw())


def export_qpy(name_root, qc_ideal, qc_transpiled, args):
    out_dir = Path(args.basePath) / "qpy"
    out_dir.mkdir(parents=True, exist_ok=True)
    ideal_path = out_dir / f"{name_root}.ideal.qpy"
    transp_path = out_dir / f"{name_root}.noisy.qpy"
    with open(ideal_path, "wb") as fd:
        qpy.dump(qc_ideal, fd)
    with open(transp_path, "wb") as fd:
        qpy.dump(qc_transpiled, fd)
    print(f"Saved QPY circuits: {ideal_path}")
    print(f"Saved QPY circuits: {transp_path}")


def collect_counts(result, n_circuits, n_qubits, verb):
    meas_log = np.zeros((2**n_qubits, n_circuits), dtype=np.int64)
    for ic in range(n_circuits):
        counts = result.get_counts(ic)
        for bitstr, count in counts.items():
            meas_log[int(bitstr.replace(" ", ""), 2), ic] = count
        if verb > 1:
            print(f"CIRCUIT {ic}: {counts}")
    return meas_log


def print_summary(meas_log, backend_name):
    n_circ = meas_log.shape[1]
    counts_per_state = meas_log.sum(axis=1)
    n_total_shots = int(counts_per_state.sum())
    n0 = int(counts_per_state[0])
    p0 = n0 / n_total_shots if n_total_shots else 0.0
    err_p0 = np.sqrt(p0 * (1.0 - p0) / n_total_shots) if n_total_shots else 0.0

    state_width = int(np.log2(meas_log.shape[0]))
    labels = [format(i, f"0{state_width}b") for i in range(len(counts_per_state))]
    counts_txt = "  ".join(f"{lab}: {int(cnt)}" for lab, cnt in zip(labels, counts_per_state))
    zero_lab = labels[0]

    print(
        "--- AGGREGATED RESULTS (sum across %d RB circuits, total shots %d, backend %s):"
        % (n_circ, n_total_shots, backend_name)
    )
    print(
        "  raw counts:  [%s]  P(%s-state):  %.4f +/- %.4f"
        % (counts_txt, zero_lab, p0, err_p0)
    )


def save_measurements(name_root, rb_tableaus, meas_log, input_path, rb_meta, noise_conf, args):
    out_dir = Path(args.basePath) / "meas"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name_root}.{args.rbType}.meas.npz"
    meta = {
        "input_rb_path": str(input_path),
        "input_rb_name": Path(str(input_path)).name,
        "rb_type": args.rbType,
        "num_sequences": int(rb_tableaus.shape[0]),
        "sequence_length": int(rb_tableaus.shape[1]),
        "shots_per_sequence": args.totShot,
        "no_barrier": bool(args.noBarrier),
        "noise_conf": noise_conf,
        "source_meta": rb_meta,
    }
    data = {
        "rb_tableaus": rb_tableaus,
        "measLog": meas_log,
    }
    write_data_npz(data, str(out_path), metaD=meta, verb=max(args.verb, 1))
    print(f"Saved measurements: {out_path}")


def main():
    args = parse_args()
    input_path, rb_tableaus, rb_meta = load_rb_tableaus(args)
    qc_ideal, n_qubits = build_rb_circuits(rb_tableaus, args.rbType, args.noBarrier)
    noise_model, noise_conf = build_noise_model(args)
    backend, qc_transpiled = transpile_circuits(qc_ideal, n_qubits, noise_model)

    if args.verb > 0:
        print(f"Loaded RB file: {input_path}")
        print(
            "RB payload: nSeq=%d seqLen=%d nq=%d"
            % (rb_tableaus.shape[0], rb_tableaus.shape[1], n_qubits)
        )
        print(f"Noise model: {noise_conf}")
        print(f"First ideal gates: {qc_ideal[0].count_ops()}")
        print(f"First transpiled gates: {qc_transpiled[0].count_ops()}")

    print_requested_circuits(args, qc_ideal[0], qc_transpiled[0])

    name_root = args.expName or f"toy_{Path(args.inputRB).stem}_{args.rbType}"
    if args.exportQPY:
        export_qpy(name_root, qc_ideal, qc_transpiled, args)

    result = backend.run(qc_transpiled, shots=args.totShot).result()
    meas_log = collect_counts(result, len(qc_transpiled), n_qubits, args.verb)
    print_summary(meas_log, backend.name)
    save_measurements(name_root, rb_tableaus, meas_log, input_path, rb_meta, noise_conf, args)


if __name__ == "__main__":
    main()
