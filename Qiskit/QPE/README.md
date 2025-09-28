Qiskit QPE demos in this directory

This folder contains three scripts that demonstrate Quantum Phase Estimation (QPE) workflows using Qiskit. Two QPE variants are provided: one for a synthetic multi‑controlled phase unitary and one for a Heisenberg‑type Hamiltonian with trotterized evolution. A utility script builds/analyses Hamiltonians and prepares eigenstates.

Overview
- prep_Heisenberg_Hamiltonian.py: Build toy Heisenberg Hamiltonians (2/3/4 qubits), compute eigenpairs classically, prepare the chosen eigenstate as a circuit (from |0…0|), and verify the eigenstate property.
- qpe_MCZ_phi.py: QPE for a synthetic unitary U = multi‑controlled Z with phase φ. Prepares a suitable computational‑basis eigenstate, performs QPE, predicts the expected n‑bit key, and saves a histogram PNG of the counts with the expected key highlighted.
- qpe_Heisenberg_Hamiltonian.py: QPE for a Heisenberg‑type Hamiltonian using PauliEvolutionGate with Suzuki‑Trotter synthesis. Selects a true eigenstate (by index), runs QPE, predicts the expected key, and saves a histogram PNG.

Requirements
- Python 3.10+
- qiskit, qiskit-aer, qiskit-ibm-runtime, numpy, scipy, matplotlib

Ensure the output directory exists before running (used for histograms):
```bash
mkdir -p out
```

Usage

1) Heisenberg utilities (build/inspect/prepare eigenstates)
```bash
./prep_Heisenberg_Hamiltonian.py --nq_ham 3 --evIdx 0
```
- --nq_ham: number of qubits in H (2, 3, or 4 supported)
- --evIdx: eigenstate index (0 = ground state)

2) QPE with synthetic MCZ(φ) unitary
```bash
./qpe_MCZ_phi.py -q 4 -x 1.28 -s 2               # repeat base gate 2^k times
./qpe_MCZ_phi.py -q 4 -x 1.28 -s 2 -p            # scale phase by 2^k (usePower)
```
- -q/--nq_phase: number of phase qubits
- -x/--true_phase: phase angle φ in radians
- -s/--nq_system: number of system qubits for U
- -p/--usePower: if set, use a single gate with phase scaled by 2^k; otherwise repeat the base gate 2^k times
- --shots: sampler shots (default 10000)

Outputs:
- Prints expected_key (most‑likely n‑bit string) and basic stats
- Saves histogram to out/qs<nq_system>_ek<expected_key>.png

3) QPE with Heisenberg Hamiltonian
```bash
./qpe_Heisenberg_Hamiltonian.py -s 3 -q 5 -t 1.2 --trotterSteps 10 --shots 10000
```
- -s/--nq_system: number of Hamiltonian qubits (2, 3, or 4 supported by the builder)
- -q/--nq_phase: number of phase qubits
- -t/--time: evolution time per base layer
- --trotterSteps: Suzuki‑Trotter repetitions per layer
- --shots: sampler shots
- --isEigen: verify the prepared eigenstate against H and exit
- -v/--verb: verbosity; if >1, print full circuit

Outputs:
- Prints expected_key (most‑likely n‑bit string) and basic stats
- Saves histogram to out/hs<nq_system>_ek<expected_key>.png

Notes
- QPE reports phases as the unitless fraction x = φ / (2π) mod 1. The printed “true” and “reco” values are fractions of 2π, not radians.
- The demos use AerSimulator with the Qiskit Runtime Sampler interface.

