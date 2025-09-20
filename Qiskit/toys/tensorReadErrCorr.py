#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Readout error correction using tensored calibration matrices for multi-qubit systems

Not implemented:
- forcing the same qubits to be calibrated as for the circuit of interest
- revert order : first principle circ of interest, then append calibration circuits
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler 
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit_aer import AerSimulator
from qiskit import transpile
from pprint import pprint

# ----------------------------------------------------------------
# Connect to IBM Quantum
# ----------------------------------------------------------------
service = QiskitRuntimeService()
backend = service.backend("ibm_fez")
backend = AerSimulator.from_backend(backend) # overwrite ideal-backend

# ----------------------------------------------------------------
# Variable number of qubits
# ----------------------------------------------------------------
nq = 3  # Change this to any number you want

# ----------------------------------------------------------------
# Build a GHZ state circuit
# ----------------------------------------------------------------
def ghz_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc

ghz = ghz_circuit(nq)

# ----------------------------------------------------------------
# Build Tensored Calibration Circuits Manually
# ----------------------------------------------------------------
def build_tensored_calibration_circuits(n_qubits):
    """Build calibration circuits for independent single-qubit calibration"""
    circuits = []
    for qubit in range(n_qubits):
        # |0> state calibration for this qubit
        qc0 = QuantumCircuit(n_qubits)
        qc0.measure_all()
        circuits.append(qc0)
        
        # |1> state calibration for this qubit
        qc1 = QuantumCircuit(n_qubits)
        qc1.x(qubit)
        qc1.measure_all()
        circuits.append(qc1)
    return circuits

cal_circuits = build_tensored_calibration_circuits(nq)
print(f"First calibration circuit:\n{cal_circuits[0]}")

# ----------------------------------------------------------------
# Run calibration circuits
# ----------------------------------------------------------------
shots = 10_000  # Increased for better statistics
options = SamplerOptions()
options.default_shots = shots

print(f"Running {len(cal_circuits)} calibration circuits on {backend.name}...")
sampler = Sampler(mode=backend, options=options)
cal_job = sampler.run(cal_circuits)
jobRes = cal_job.result()

nCirc = len(jobRes)  # number of circuits in the job    
countsL = [jobRes[i].data.meas.get_counts() for i in range(nCirc)]

print('Sample calibration counts:')
pprint(countsL[:2])  # Show first 2 for brevity

# ----------------------------------------------------------------
# Build tensored calibration matrix from results
# ----------------------------------------------------------------
def build_tensored_cal_matrix(cal_counts, n_qubits, shots):
    """Build tensored calibration matrix from independent qubit calibrations"""
    # Initialize single-qubit calibration matrices
    qubit_cal_matrices = []
    
    for qubit in range(n_qubits):
        # Results for this qubit's calibration circuits
        idx_0 = 2 * qubit      # |0> calibration
        idx_1 = 2 * qubit + 1  # |1> calibration
        
        # Extract counts
        counts_0 = cal_counts[idx_0]
        counts_1 = cal_counts[idx_1]
        
        # Build 2x2 calibration matrix for this qubit
        # M[i,j] = P(measure i | prepare j)
        M = np.zeros((2, 2))
        
        # When preparing |0>, probability of measuring 0 or 1
        for bitstring, count in counts_0.items():
            # bitstring is a string like '000' or '001'
            # Extract the bit for this qubit (counting from right)
            bit = int(bitstring[n_qubits - 1 - qubit])
            M[bit, 0] += count / shots
            
        # When preparing |1>, probability of measuring 0 or 1  
        for bitstring, count in counts_1.items():
            bit = int(bitstring[n_qubits - 1 - qubit])
            M[bit, 1] += count / shots
            
        qubit_cal_matrices.append(M)
    
    # Build full tensored matrix
    # Using Kronecker product
    full_matrix = qubit_cal_matrices[0]
    for M in qubit_cal_matrices[1:]:
        full_matrix = np.kron(full_matrix, M)
    
    return full_matrix, qubit_cal_matrices

A, qubit_matrices = build_tensored_cal_matrix(countsL, nq, shots)

# Print single-qubit error rates
print("\nSingle-qubit readout error rates:")
print("-" * 40)
for i, M in enumerate(qubit_matrices):
    err_0to1 = 1 - M[0, 0]  # P(measure 1 | prepare 0)
    err_1to0 = 1 - M[1, 1]  # P(measure 0 | prepare 1)
    print(f"Qubit {i}: ε(0→1) = {err_0to1:.4f}, ε(1→0) = {err_1to0:.4f}")

# ----------------------------------------------------------------
# Run GHZ circuit using Sampler
# ----------------------------------------------------------------
print("\nRunning GHZ circuit...")
sampler = Sampler(mode=backend, options=options)
qcT =  transpile(ghz, backend,optimization_level=3)
ghz_job = sampler.run([qcT])
ghz_result = ghz_job.result()

# Extract counts for GHZ circuit
ghz_counts = ghz_result[0].data.meas.get_counts()
print(f"\nRaw GHZ counts (top entries):")
sorted_counts = sorted(ghz_counts.items(), key=lambda x: x[1], reverse=True)
for state, count in sorted_counts[:5]:
    print(f"  {state}: {count}")

# ----------------------------------------------------------------
# Convert counts to probability vector
# ----------------------------------------------------------------
n_states = 2**nq
p_uncorr = np.zeros(n_states)

for bitstring, count in ghz_counts.items():
    # Convert bitstring to integer index
    idx = int(bitstring, 2)
    p_uncorr[idx] = count / shots

# Statistical errors (binomial approximation)
err_uncorr = np.sqrt(np.maximum(p_uncorr * (1 - p_uncorr) / shots, 0))

# ----------------------------------------------------------------
# Apply mitigation
# ----------------------------------------------------------------
# Apply mitigation: p_corr = A^{-1} p_uncorr
A_inv = np.linalg.pinv(A)  # Use pseudo-inverse for stability
p_corr = A_inv @ p_uncorr

# Ensure probabilities are normalized and non-negative
p_corr = np.maximum(p_corr, 0)  # Clip negative values
p_corr = p_corr / np.sum(p_corr)  # Renormalize

# Propagate statistical errors: Cov_corr = A^{-1} Cov_uncorr (A^{-1})^T
Cov_uncorr = np.diag(err_uncorr**2)
Cov_corr = A_inv @ Cov_uncorr @ A_inv.T
err_corr = np.sqrt(np.abs(np.diag(Cov_corr)))  # abs to handle numerical issues

# ----------------------------------------------------------------
# Print results for significant states only
# ----------------------------------------------------------------
print("\n" + "="*70)
print("RESULTS (showing states with p > 0.01)")
print("="*70)
print(f"{'State':<10} | {'Uncorrected p ± σ':<25} | {'Corrected p ± σ':<25}")
print("-"*70)

for i in range(n_states):
    if p_uncorr[i] > 0.01 or p_corr[i] > 0.01:
        state = format(i, f'0{nq}b')  # Format with correct number of bits
        print(f"{state:<10} | {p_uncorr[i]:>8.6f} ± {err_uncorr[i]:.6f} | "
              f"{p_corr[i]:>8.6f} ± {err_corr[i]:.6f}")

# ----------------------------------------------------------------
# Compute GHZ fidelity metrics
# ----------------------------------------------------------------
print("\n" + "="*70)
print("GHZ STATE FIDELITY METRICS")
print("="*70)

# Ideal GHZ state has only |00...0> and |11...1>
all_zeros = 0
all_ones = (1 << nq) - 1  # 2^nq - 1

p_all0_uncorr = p_uncorr[all_zeros]
p_all1_uncorr = p_uncorr[all_ones]
p_all0_corr = p_corr[all_zeros]
p_all1_corr = p_corr[all_ones]

err_all0_uncorr = err_uncorr[all_zeros]
err_all1_uncorr = err_uncorr[all_ones]
err_all0_corr = err_corr[all_zeros]
err_all1_corr = err_corr[all_ones]

ghz_fidelity_uncorr = p_all0_uncorr + p_all1_uncorr
ghz_fidelity_corr = p_all0_corr + p_all1_corr

# Error in fidelity (assuming independent errors)
ghz_fidelity_err_uncorr = np.sqrt(err_all0_uncorr**2 + err_all1_uncorr**2)
ghz_fidelity_err_corr = np.sqrt(err_all0_corr**2 + err_all1_corr**2)

all0_str = '0' * nq
all1_str = '1' * nq

print(f"P(|{all0_str}>) uncorrected: {p_all0_uncorr:.6f} ± {err_all0_uncorr:.6f}")
print(f"P(|{all1_str}>) uncorrected: {p_all1_uncorr:.6f} ± {err_all1_uncorr:.6f}")
print(f"P(|{all0_str}>) corrected:   {p_all0_corr:.6f} ± {err_all0_corr:.6f}")
print(f"P(|{all1_str}>) corrected:   {p_all1_corr:.6f} ± {err_all1_corr:.6f}")
print(f"\nGHZ Fidelity (uncorrected): {ghz_fidelity_uncorr:.4f} ± {ghz_fidelity_err_uncorr:.4f}")
print(f"GHZ Fidelity (corrected):   {ghz_fidelity_corr:.4f} ± {ghz_fidelity_err_corr:.4f}")

# Error amplification factor
valid_indices = (err_uncorr > 1e-10)
if np.any(valid_indices):
    avg_err_amplification = np.mean(err_corr[valid_indices] / err_uncorr[valid_indices])
    print(f"\nAverage error amplification from mitigation: {avg_err_amplification:.2f}x")

# ----------------------------------------------------------------
# Overhead summary
# ----------------------------------------------------------------
print("\n" + "="*70)
print("OVERHEAD SUMMARY")
print("="*70)
full_cal_circuits = 2**nq
print(f"Calibration circuits: {len(cal_circuits)} (vs {full_cal_circuits} for full calibration)")
print(f"Calibration shots: {len(cal_circuits) * shots:,} (vs {full_cal_circuits * shots:,} for full)")
print(f"Reduction factor: {full_cal_circuits/len(cal_circuits):.1f}x")
