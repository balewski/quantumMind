# SQUIN Toy Programs

These scripts exercise Bloqade SQUIN kernels with PyQrack
`DynamicMemorySimulator`. The simulator path used here is state-vector based;
shot statistics are produced by repeatedly calling `task.run()` one shot at a
time.

## Files

- `bell1.py`
  - Ideal 2-qubit Bell-state sampler.
  - Builds `H(0); CX(0,1); measure`.
  - Prints Cirq circuit at `--verb 1`, SQUIN IR at `--verb 2`.

- `ghz1.py`
  - Ideal n-qubit GHZ sampler.
  - Uses `-q/--qubits` to choose the GHZ size.
  - Reuses generic shot counting and circuit printing from `Util_Squin.py`.

- `ghz2.py`
  - Noisy n-qubit GHZ sampler.
  - Adds `--noise_1q`, `--noise_2q`, `--noise_erasure`, and `--noise_readout`.
  - Erasure is applied after each 2-qubit gate checkpoint.
  - Splits final counts into shots with and without detected erasure markers.

- `ghz3.py`
  - Bell-state sampler using Bloqade's Gemini Cirq noise pipeline.
  - Converts SQUIN to Cirq, applies `--zone_type 1` or `--zone_type 2`, then
    loads the noisy circuit back into SQUIN for PyQrack sampling.
  - `--atom_loss` is implemented as one explicit SQUIN qubit-loss checkpoint
    before measurement because Gemini Cirq models do not currently inject atom loss.

- `qft1.py`
  - Ideal inverse-QFT decoding benchmark.
  - Manually prepares the Fourier product state for integer `-k/--freq`.
  - Applies inverse QFT and reports target-state fidelity and top leakage states.

- `qft2.py`
  - Noisy inverse-QFT benchmark.
  - Uses the same noise knobs as `ghz2.py`.
  - Splits counts by erasure and computes fidelity using only no-erasure shots.

- `Util_Squin.py`
  - Shared utilities for the toy scripts.
  - Provides circuit printing, one-shot PyQrack sampling, bitstring conversion,
    loss/erasure summaries, probability validation, and count formatting.

- `dump_noiseModel.py`
  - Standalone inspector for Gemini one- and two-zone noise models.
  - Uses `--zone_type` to select the model and `--atom_loss` to rewrite all
    loss-like parameters before dumping public fields.
  - Exports helper functions reused by `ghz3.py`.

## Common Examples

```bash
./bell1.py --shots 1000
./ghz1.py -q 5 --shots 1000
./ghz2.py -q 5 --noise_1q 0.01 --noise_2q 0.02 --noise_erasure 0.03
./ghz3.py --zone_type 1 --atom_loss 0.05
./qft1.py -q 3 -k 5 -n 2000
./qft2.py -q 3 -k 5 --noise_1q 0.01 --noise_2q 0.02 --noise_readout 0.01
./dump_noiseModel.py --zone_type 2 --atom_loss 0.02
```
