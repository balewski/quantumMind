#!/usr/bin/env python3
"""
This program benchmarks the fidelity of the Quantum Fourier Transform (QFT) by manually encoding an integer 'k' into qubit phases using precise single-qubit gates.
It then executes the Inverse QFT circuit to decode this "Fourier state" back into the binary representation of 'k'.
By measuring how often the output correctly matches 'k', the script calculates the fidelity of the QFT circuit isolated from state preparation errors.
The results include the success probability (fidelity), error rate, and a list of incorrect "leakage" states to help diagnose specific hardware faults.
This method is ideal for noisy devices because the state preparation is cheap and accurate, making the QFT the primary source of measured error.
"""

import argparse
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFTGate
 
def post_process_counts(counts, args):
    """Post-process a counts dictionary: print fidelity, infidelity and top outcomes."""
    print("\n--- Post-processing counts ---")
    target_str = f"{args.freq:0{args.qubits}b}"
    success_count = counts.get(target_str, 0)
    fidelity = success_count / args.shots if args.shots else 0.0
    infidelity = 1.0 - fidelity
    print(f"Target State: |{target_str}>")
    print(f"Success Count: {success_count} / {args.shots}")
    print(f"Fidelity:   {fidelity:.4f}")
    print(f"Infidelity: {infidelity:.4f}")
    print("\nTop measured states:")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for state, count in sorted_counts[:5]:
        mark = " (TARGET)" if state == target_str else ""
        print(f"  |{state}> : {count}{mark}")

def create_manual_freq_test(num_qubits, k):
    """
    1. Manually prepares the Fourier state |~k> using H and P gates.
       (This represents a signal with frequency 'k').
    2. Runs the Inverse QFT circuit to decode it.
    3. If QFT is perfect, we measure |k>.
    """
    qc = QuantumCircuit(num_qubits)
    
    # --- 1. Manual State Preparation (The "Input") ---
    # We create the state that QFT would produce if the input were |k>.
    # |~k> is a product state where qubit j has a specific phase rotation.
    # Formula (Little Endian): Phase(j) = 2*pi * k / 2^(num_qubits - j)
    
    qc.h(range(num_qubits)) # Start with equal superposition
    
    print(f"Preparing Fourier state |~{k}> on {num_qubits} qubits:")
    for j in range(num_qubits):
        # Calculate angle: 2*pi * k * 2^j / 2^N
        # Simplifies to: 2*pi * k / 2^(N-j)
        angle = 2 * np.pi * k / (2**(num_qubits - j))
        qc.p(angle, j)
        print( f"  Qubit {j}: phi={angle:.4f} rad ")
        
    qc.barrier()
    
    # --- 2. The Device Under Test: Inverse QFT ---
    # We use Inverse QFT to map the Fourier Basis back to Computational Basis.
    # This has the exact same complexity/depth as standard QFT.
    # QFTGate does not accept `do_swaps` in some Qiskit versions; omit it.
    # inverse() already returns a Gate/Instruction, so no .to_gate() is needed.
    iqft_gate = QFTGate(num_qubits).inverse()
    iqft_gate.label = "InvQFT"
    qc.append(iqft_gate, range(num_qubits))
    
    qc.measure_all()
    return qc

def main():
    parser = argparse.ArgumentParser(description="QFT Fidelity Benchmark")
    parser.add_argument('-q', '--qubits', type=int, default=3, help="Number of qubits")
    parser.add_argument('-k', '--freq', type=int, default=1, help="Input Frequency to test")
    parser.add_argument('-n', '--shots', type=int, default=2000, help="Number of shots")
    parser.add_argument('-c', "--printIdealCirc", action="store_true", help="Print circuit")
    parser.add_argument('-C', "--printTranspCirc", action="store_true", help="Print circuit")
    parser.add_argument('-b', '--backend', default='ideal', help="quantum backend")
    args = parser.parse_args()

    # 1. Build Circuit
    qc = create_manual_freq_test(args.qubits, args.freq)

    # 2. Backend selection and transpilation target
    if args.backend == 'ideal':
        print(f"\nSelected backend: ideal (AerSimulator). Transpiling for local simulator...")
        backend = AerSimulator()
        qcT = transpile(qc, backend, optimization_level=3, basis_gates=['u','h','p','cz'])        
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        print(f"\nSelected backend: {args.backend} (remote). Transpiling for device runtime...")
        service = QiskitRuntimeService()
        backend = service.backend(args.backend)
        qcT = transpile(qc, backend, optimization_level=3)        

    print(f"--- Benchmarking QFT (Size={args.qubits}) ---")
    if args.printIdealCirc:        print(qc)
    if args.printTranspCirc:
        print("\n--- Transpiled circuit for selected backend (qcT) ---")
        print(qcT)

    print('transp gates:',qcT.count_ops())
    print(f"Input Frequency: {args.freq}")

    # 3. Always use Sampler for execution (local or remote)
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    print(f'\nRunning sampler on {backend.name} nq={args.qubits}...')
    sampler = Sampler(mode=backend)
    job = sampler.run([qcT], shots=args.shots)
    res = job.result()

    # Extract counts using the SDK-recommended path (no fallbacks)
    counts = res[0].data.meas.get_counts()
    print('counts:', counts)
    post_process_counts(counts, args)
    return

if __name__ == "__main__":
    main()
