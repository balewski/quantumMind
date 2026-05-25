# RB Tableau

This directory contains two scripts:

- `gen_RB_seq.py`: generate randomized benchmarking (RB) Clifford sequences
- `run_one_noisy_RB.py`: run those sequences on `qiskit-aer` with a simple noise model

## RB math used here

For an `n`-qubit RB experiment, each sequence starts in `|0...0>`, applies a product of random Clifford gates, and ends with an inversion Clifford chosen so that the ideal final state returns to `|0...0>`.

For standard RB, one sequence is

```text
C_inv C_m ... C_2 C_1 |0...0>
```

where each `C_k` is a random element of the `n`-qubit Clifford group and

```text
C_inv = (C_m ... C_2 C_1)^(-1)
```

In the ideal noiseless case, the whole product is the identity, so measuring in the computational basis should give the all-zero bitstring with probability 1.

For interleaved RB, the code inserts a target gate `G_k` between random Cliffords:

```text
C_inv G_m C_m ... G_2 C_2 G_1 C_1 |0...0>
```

and chooses

```text
C_inv = (G_m C_m ... G_2 C_2 G_1 C_1)^(-1)
```

Again, the ideal net action is the identity. Under noise, the survival probability of the all-zero state decays with sequence depth, which is the RB signal.

## Supported sequence types

`gen_RB_seq.py` supports:

- `std1q`: standard 1-qubit RB
- `std2q`: standard 2-qubit RB
- `std3q`: standard 3-qubit RB
- `int2q`: interleaved 2-qubit RB with a randomly chosen directed `CX`, either `0->1` or `1->0`
- `int3q`: interleaved 3-qubit RB with a randomly chosen directed `CX` from all 6 control-target pairs

Not supported:

- Toffoli / `CCX`, because it is not a Clifford gate and therefore does not belong in this Clifford-tableau RB flow

## How this generator defines sequence length

The CLI option `--seqLen M` means the logical RB length is `M`.

- Standard RB generates `M-1` random Cliffords and then appends one inversion Clifford
- Interleaved RB generates `M-1` random Cliffords, inserts one interleaved target after each random Clifford, and then appends one inversion Clifford

So the stored circuit depth in Clifford steps differs between standard and interleaved RB, but both are organized around the same logical RB length parameter `M`.

## Generate RB sequences

Run from the repository root so the local `toolbox/` module is importable.

Example: standard 2-qubit RB

```bash
cd /Users/balewski/shared_volumes/quantumMind
python3 Qiskit/RB_tableau/gen_RB_seq.py --rbType std2q --numSeq 20 --seqLen 16 --seed 123
```

Example: interleaved 3-qubit RB

```bash
python3 Qiskit/RB_tableau/gen_RB_seq.py --rbType int3q --numSeq 20 --seqLen 16 --seed 123
```

Useful options:

- `--verify 1`: reconstruct each sequence, simulate it ideally, and check that the final state returns to `|0...0>`
- `--printCirc i`: print one ideal example circuit
- `--printCirc d`: print one decomposed circuit

## Run noisy Qiskit on one RB set

`run_one_noisy_RB.py` rebuilds circuits from the saved Clifford tableaus, transpiles them for Aer, attaches a simple noise model, and measures all qubits in the computational basis.

The noise model in this script can include:

- 1-qubit depolarizing error
- 2-qubit depolarizing error on `cx`
- readout error
- reset error
- optional idle-gate depolarizing error on `id`

Example: run noisy standard 2-qubit RB

```bash
cd /Users/balewski/shared_volumes/quantumMind
python3 Qiskit/RB_tableau/run_one_noisy_RB.py \
  --rbType std2q \
  --inputRB rbseq_m16_20 \
  --totShot 4000 \
  --noise_err1Q 1e-4 \
  --noise_err2Q 4e-4 \
  --noise_errRead 2e-4
```

Example: noisy interleaved 2-qubit RB

```bash
python3 Qiskit/RB_tableau/run_one_noisy_RB.py \
  --rbType int2q \
  --inputRB rbseq_m16_20 \
  --totShot 4000 \
  --noise_err1Q 1e-4 \
  --noise_err2Q 8e-4 \
  --noise_errRead 5e-4
```

What the runner does:

1. Reconstruct each Clifford sequence as a `QuantumCircuit`
2. Append measurements in the computational basis
3. Transpile to Aer basis gates
4. Execute all RB circuits with the requested shot count
5. Report the aggregated probability of the all-zero outcome

In the ideal case, that probability should be 1. With noise enabled, it drops below 1 and becomes the observable RB survival signal.

## Requirements

At minimum, these scripts expect:

- `python3`
- `numpy`
- `qiskit`
- `qiskit-aer`
