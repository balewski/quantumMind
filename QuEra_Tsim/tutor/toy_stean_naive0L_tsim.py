#!/usr/bin/env python3
"""
Stand-alone version of encoding_0L_Steane.ipynb.

Encodes a noisy logical |0> state with the [[7,1,3]] Steane color code,
compares raw vs. MWPM vs. Tesseract decoders at a single p_err, then runs a
sinter Monte Carlo sweep over a range of physical error rates and saves the
plot.

Outputs (in out/):
  - circuit_<hash>.svg        the noisy encoding circuit diagram
  - decoder_sweep_<hash>.png  the MC sweep plot
"""

import argparse
import hashlib,os
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import sinter
from tsim import Circuit
from tesseract_decoder import tesseract, TesseractSinterDecoder

from utils.binomial_standard_error import binomial_standard_error as stat_err
from utils.no_decoder import NoDecoder


def make_circuit(p):
    return Circuit(f"""
        R 0 1 2 3 4 5 6
        TICK
        SQRT_Y_DAG 0 1 2 3 4 5
        DEPOLARIZE1({p}) 0 1
        TICK
        CZ 1 2 3 4 5 6
        DEPOLARIZE2({p}) 1 2
        #T 0  # insert artifical error
        TICK
        SQRT_Y 6
        DEPOLARIZE1({p}) 6
        TICK
        CZ 0 3 2 5 4 6
        TICK
        SQRT_Y 2 3 4 5 6
        #T 3  # insert artifical error
        DEPOLARIZE1({p}) 2 4 6
        TICK
        CZ 0 1 2 3 4 5
        #T 2 3  # insert artifical error
        TICK
        DEPOLARIZE1({p}) 0 1 2 3 4 5 6
        SQRT_Y 1 2 4
        X 3
        TICK
        M 0 1 2 3 4 5 6
        DETECTOR rec[-7] rec[-6] rec[-5] rec[-4]
        DETECTOR rec[-6] rec[-5] rec[-3] rec[-2]
        DETECTOR rec[-5] rec[-4] rec[-3] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-7] rec[-6] rec[-2]
        """)


def compare_decoders(c, shots):
    det_sampler = c.compile_detector_sampler()
    det_samples, obs_samples = det_sampler.sample(shots=shots, separate_observables=True)

    dem = c.detector_error_model()
    decoder = tesseract.TesseractConfig(dem=dem).compile_decoder()

    obs_tesseract = np.zeros_like(obs_samples)
    for i, det_sample in enumerate(det_samples):
        flip_obs = decoder.decode(det_sample)
        obs_tesseract[i] = np.logical_xor(obs_samples[i], flip_obs[0])

    mwpm_predictions = sinter.predict_observables(
        dem=dem, dets=det_samples, decoder="pymatching", bit_pack_result=False,
    )
    obs_mwpm = np.logical_xor(obs_samples, mwpm_predictions)

    N = len(obs_samples)
    n_raw = int(np.count_nonzero(obs_samples))
    n_mwpm = int(np.count_nonzero(obs_mwpm))
    n_tess = int(np.count_nonzero(obs_tesseract))

    print(f"\n--- Single-point comparison (Nshots={N}) ---")
    print(f"Raw obs.:  {n_raw/N:.4f} +/- {stat_err(n_raw, N):.4f}")
    print(f"MWPM:      {n_mwpm/N:.4f} +/- {stat_err(n_mwpm, N):.4f}")
    print(f"Tesseract: {n_tess/N:.4f} +/- {stat_err(n_tess, N):.4f}")


def run_sweep(noise_range, shots_per_point):
    lo, hi, n = noise_range
    noises = np.logspace(lo, hi, int(n))

    # Sample raw measurements / detectors / observables for each noise point.
    # First index is noise; remaining dims are (shots, qubits|detectors|observables).
    meas_arr, det_arr, obs_arr = [], [], []
    for p in noises:
        c = make_circuit(p)
        
        # 1. Sample the raw measurements ONCE
        m_sampler = c.compile_sampler()
        meas_samples = m_sampler.sample(shots=shots_per_point)
        
        # 2. Deterministically convert those exact measurements into detectors and observables
        converter = c.compile_m2d_converter()
        det_samples, obs_samples = converter.convert(
            measurements=meas_samples, 
            separate_observables=True
        )
        
        meas_arr.append(meas_samples)
        det_arr.append(det_samples)
        obs_arr.append(obs_samples)
        
    meas_arr = np.array(meas_arr)
    det_arr = np.array(det_arr)
    obs_arr = np.array(obs_arr)

    tasks = [
        sinter.Task(circuit=make_circuit(p).cast_to_stim(), json_metadata={"p": p})
        for p in noises
    ]

    collected = sinter.collect(
        num_workers=1,
        tasks=tasks,
        decoders=["tesseract", "no decoding"],
        max_shots=shots_per_point,
        max_errors=shots_per_point,
        custom_decoders={
            "tesseract": TesseractSinterDecoder(),
            "no decoding": NoDecoder(),
        },
        start_batch_size=shots_per_point,
        max_batch_size=shots_per_point,
    )
    mwpm = sinter.collect(
        num_workers=1,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=shots_per_point,
        max_errors=shots_per_point,
        start_batch_size=shots_per_point,
        max_batch_size=shots_per_point,
    )
    return collected + mwpm, noises, meas_arr, det_arr, obs_arr


def make_plot(stats, shots_per_point, png_path, h):
    def curve(decoder_name):
        pts = [(s.json_metadata["p"], s.errors, s.shots)
               for s in stats if s.decoder == decoder_name]
        pts.sort()
        x = np.array([p for p, _, _ in pts])
        y = np.array([n / N for _, n, N in pts])
        err = np.array([stat_err(n, N) for _, n, N in pts])
        return x, y, err

    fig, ax = plt.subplots(1, 1)
    label_map = {"pymatching": "MWPM", "tesseract": "Tesseract", "no decoding": "No decoding"}
    color_map = {"pymatching": "#1f77b4", "tesseract": "#ff7f0e", "no decoding": "#2ca02c"}
    marker_map = {"pymatching": "o", "tesseract": "s", "no decoding": "o"}
    for decoder_name, label in label_map.items():
        x, y, err = curve(decoder_name)
        if len(x) == 0:
            continue
        color = color_map[decoder_name]
        plot_kwargs = {
            "marker": marker_map[decoder_name],
            "label": label,
            "color": color,
        }
        if decoder_name == "no decoding":
            plot_kwargs["markerfacecolor"] = "none"
            plot_kwargs["markeredgewidth"] = 1.5
        ax.plot(x, y, **plot_kwargs)
        ax.fill_between(x, y - err, y + err, alpha=0.25, color=color)

    ax.loglog()
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel(r"Probability of logical $|\bar{1}\rangle$")
    ax.set_title(f"Tsim(QuEra): Prep |0L> meas |1L>"
                 f"(Nshots/point = {shots_per_point})  [{h}]")
    ax.set_ylim(bottom=1e-4)
    ax.grid()
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0)
    x_left = min(s.json_metadata["p"] for s in stats) * 1.05
    ax.text(x_left, 0.48, "random", color="red", ha="left", va="top")
    ax.legend(title="Decoder")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p_err", type=float, default=0.01,
                        help="physical error rate for single-point comparison")
    parser.add_argument("--shots", type=int, default=1024 * 4,
                        help="shots per point (single-point and per MC point)")
    parser.add_argument("--noise_range", type=float, nargs=3,
                        default=[-3.2, -0.2, 6],
                        metavar=("LOG10_LO", "LOG10_HI", "N"),
                        help="np.logspace args for the MC sweep")
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    h = hashlib.md5(os.urandom(16)).hexdigest()[:6]

    c = make_circuit(args.p_err)

    svg_path = out_dir / f"circuit_{h}.svg"
    svg_path.write_text(str(c.diagram("timeline-svg", height=350)))
    print(f"Saved: {svg_path}")

    compare_decoders(c, shots=args.shots)

    T0 = time()
    stats, noises, meas_arr, det_arr, obs_arr = run_sweep(args.noise_range, args.shots)
    print(f"noise scan elaT={time()-T0:.1f}sec")

    make_plot(stats, args.shots, out_dir / f"decoder_sweep_{h}.png", h)

    print(f"\n--- Stored arrays ---")
    print(f"noise:        {noises.shape}  dtype={noises.dtype}  values={noises}")
    print(f"measurements: {meas_arr.shape}  dtype={meas_arr.dtype}")
    print(f"detectors:    {det_arr.shape}  dtype={det_arr.dtype}")
    print(f"observables:  {obs_arr.shape}  dtype={obs_arr.dtype}")


if __name__ == "__main__":
    main()
