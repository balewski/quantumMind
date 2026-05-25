#!/usr/bin/env python3
"""
=======================================================
Randomized Benchmarking (RB) generator
=======================================================

Generates:
  - std1q : Standard 1-qubit RB
  - std2q : Standard 2-qubit RB
  - std3q : Standard 3-qubit RB
  - int2q : Interleaved 2-qubit RB (Random Target: CNOT 0->1 or 1->0)
  - int3q : Interleaved 3-qubit RB (Random Target: Any directed CNOT in 3Q)

  * NOTE: Toffoli (CCX) is NOT supported because it is not a Clifford gate.

Output file name: rb_m[seqLen]_[numSeq].[rbType].npz
"""

import argparse
import os
import numpy as np
from pprint import pprint
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford, random_clifford
from qiskit_aer import AerSimulator
from toolbox.Util_NumpyIOv1 import write_data_npz, read_data_npz

# Suppress qiskit transpiler logs
import logging
logging.getLogger('qiskit.passmanager').setLevel(logging.WARNING)
logging.getLogger('qiskit.compiler').setLevel(logging.WARNING)


# ====================================================
# 1. RB Sequence Generation
# ====================================================

def generate_rb_sequences(num_seq, seq_len, rbType, seed=None):
    """
    Generates RB sequences for requested rbType.
    Returns:
        sequences_cliffords: List of lists of Clifford objects (for math/simulation)
        sequences_circuits: List of lists of QuantumCircuit objects (for visualization ONLY)
        n_qubits: int
    """
    if seed is not None:
        np.random.seed(seed)

    # Defaults
    n_qubits = 0
    interleaved = False
    
    # Pools for random interleaved gates
    # We store tuples: (Clifford_Object, Circuit_Object)
    # This allows us to use the Clifford for math, but the Circuit for perfect visualization
    target_pool = [] 

    # --- Configuration Logic ---
    if rbType == "std1q":
        n_qubits = 1
    elif rbType == "std2q":
        n_qubits = 2
    elif rbType == "std3q":
        n_qubits = 3
    
    elif rbType == "int2q":
        n_qubits = 2
        interleaved = True
        # Random Pool: All directed CNOTs on 2 qubits
        # Pairs: (0,1), (1,0)
        pairs = [(0, 1), (1, 0)]
        for c, t in pairs:
            qc = QuantumCircuit(2)
            qc.cx(c, t)
            target_pool.append( (Clifford(qc), qc) )
        
    elif rbType == "int3q":
        n_qubits = 3
        interleaved = True
        # Random Pool: All 6 directed CNOTs on 3 qubits
        pairs = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
        for c, t in pairs:
            qc = QuantumCircuit(3)
            qc.cx(c, t)
            target_pool.append( (Clifford(qc), qc) )

    elif rbType == "tof3q":
        raise ValueError("Toffoli (CCX) is NOT a Clifford gate and cannot be used in Clifford RB.")
    else:
        raise ValueError(f"Unsupported rbType: {rbType}")

    # --- Generation Loop ---
    all_sequences_objs = []
    
    # We also track the "Display Circuits" to verify randomization visually
    # This list will mirror all_sequences_objs but contain Qiskit Circuits for the INTERLEAVED gates
    all_sequences_circs = []

    for _ in range(num_seq):
        seq_cliffords = []
        seq_circuits = [] # Store explicit circuits for interleaved steps
        
        # Initialize Identity Clifford using Symplectic dimensions (2*N)
        cumulative_unitary = Clifford(np.eye(2 * n_qubits, dtype=bool))

        for _ in range(seq_len):
            # 1. Random Clifford
            rc = random_clifford(n_qubits)
            seq_cliffords.append(rc)
            seq_circuits.append(None) # No explicit circuit for random steps (let verify decompose it)
            
            cumulative_unitary = cumulative_unitary.compose(rc)

            # 2. Interleaved Gate (Random selection from pool)
            if interleaved and len(target_pool) > 0:
                # Randomly pick one gate from the configured pool
                idx = np.random.randint(len(target_pool))
                target_cliff, target_circ = target_pool[idx]
                
                seq_cliffords.append(target_cliff)
                seq_circuits.append(target_circ) # Save the specific circuit (e.g. CX 1->0)
                
                cumulative_unitary = cumulative_unitary.compose(target_cliff)

        # 3. Inversion Step
        inv_clifford = cumulative_unitary.adjoint()
        seq_cliffords.append(inv_clifford)
        seq_circuits.append(None)
        
        all_sequences_objs.append(seq_cliffords)
        all_sequences_circs.append(seq_circuits)

    return all_sequences_objs, all_sequences_circs, n_qubits

# =============================================================================
# 2. Circuit Construction & Verification
# =============================================================================

def decompose_clifford_to_circuit(clifford_obj, n_qubits):
    """Decomposes a Clifford object into a QuantumCircuit."""
    qc = clifford_obj.to_circuit()
    basis = ['sx', 'rz', 'x', 'z', 'h', 'id']
    if n_qubits >= 2:
        basis = ['cx'] + basis
        
    # Optimization level 2 cleans up 'messy' synthesis
    qc_transpiled = transpile(qc, basis_gates=basis, optimization_level=2)
    return qc_transpiled


def verify_sequences(tableau_array, circuit_history, md):
    """Verifies sequences from Numpy Tableaus, using circuit_history for clean visualization."""
    num_seq = tableau_array.shape[0]
    n_qubits = md['n_qubits']
    interleaved = 'int' in md['rb_type']
    
    pprint(md)
    print(f"\n--- Verifying {num_seq} sequences (nq={n_qubits}) from TABLEAUS ---")
    
    sim = AerSimulator(method='statevector')
    success_count = 0
    
    full_qc_sample = None

    for i in range(num_seq):
        full_qc = QuantumCircuit(n_qubits)
        
        # Retrieve the explicit circuit list for this sequence
        # (This allows us to see exactly which CNOT was picked)
        seq_explicit_circs = circuit_history[i]
        
        for j in range(tableau_array.shape[1]):
            cliff = Clifford(tableau_array[i, j])
            
            # Check if we have an explicit circuit for this step (Interleaved Gate)
            # seq_explicit_circs matches the index j
            explicit_circ = seq_explicit_circs[j] if j < len(seq_explicit_circs) else None

            # Visualization Barriers
            if interleaved:
                if j % 2 == 0:
                    full_qc.barrier(label=f'R{j//2}')
                else:
                    full_qc.barrier(label='Tgt')
            else:
                full_qc.barrier(label=f'C{j}')
            
            if explicit_circ is not None:
                # Use the original explicit circuit (e.g. CX 1->0) for visualization
                # This guarantees we see what we picked, not what Clifford.to_circuit() synthesized
                full_qc.compose(explicit_circ, qubits=list(range(n_qubits)), inplace=True)
            else:
                # Use standard decomposition for Random gates
                sub_qc = decompose_clifford_to_circuit(cliff, n_qubits)
                full_qc.compose(sub_qc, qubits=list(range(n_qubits)), inplace=True)
            
        if i == 0:
            full_qc_sample = full_qc

        full_qc.save_statevector()
        result = sim.run(full_qc).result()
        sv = result.get_statevector(full_qc)
        
        prob_all0 = np.abs(sv[0])**2
        if np.isclose(prob_all0, 1.0, atol=1e-5):
            success_count += 1
        else:
            print(f"Sequence {i} FAILED. Prob(|0...0>) = {prob_all0:.4f}")

    print(f"Result: {success_count}/{num_seq} passed.")
    return full_qc_sample

# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="RB Generator")
    parser.add_argument('--numSeq', type=int, default=5, help='Number of RB sequences')
    parser.add_argument('-m','--seqLen', type=int, default=2, help='Length of random segment (M).')
    
    parser.add_argument('--rbType', 
                        choices=['std1q', 'std2q', 'std3q', 'int2q', 'int3q'], 
                        default='std2q',
                        help='RB type: std1q, std2q, std3q, int2q, int3q')
                        
    parser.add_argument('--verify', type=int, default=0, help='Verification: 0=None, 1=Run Simulator')
    parser.add_argument('-c',"--printCirc", type=str, default="", help="i=ideal, d=decomposed, t=transpiled")

    parser.add_argument("--dataName", type=str, default=None)
    parser.add_argument("--basePath", default='out', help="Output directory")
    parser.add_argument('-s',"--seed", type=int, default=None)
    parser.add_argument('-v',"--verb", type=int, default=1)

    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)

    mode_str = args.rbType
    if args.dataName is None:
        args.dataName = f'rbseq_m{args.seqLen}_{args.numSeq}'
    
    # Choose subdir by qubit count
    if '1q' in args.rbType:
        subdir = 'inputRB_1Q'
    elif '3q' in args.rbType:
        subdir = 'inputRB_3Q'
    else:
        subdir = 'inputRB_2Q'
        
    args.dataPath = os.path.join(args.basePath, subdir)
    print("Args:", vars(args))
    if not os.path.exists(args.dataPath):
        os.makedirs(args.dataPath)
 
    # 1. Generate Sequences
    print(f"\nGenerating {mode_str} sequences...")
    # UPDATED: Returns 3 values now
    sequences_objs, sequences_circs, n_qubits = generate_rb_sequences(args.numSeq, args.seqLen-1, args.rbType, args.seed)
    
    # Calculate total elements per sequence for storage
    total_len = len(sequences_objs[0])
    print(f"Total Cliffords per sequence: {total_len} (seqLen M={args.seqLen}, nq={n_qubits})")
    
    # 2. Convert to Numpy (Tableaus)
    # Shape: [numSeq, actual_total_len, 2*n, 2n+1]
    tab_shape = (args.numSeq, total_len, 2 * n_qubits, 2 * n_qubits + 1)
    tableau_storage = np.zeros(tab_shape, dtype=bool)
    
    for i, seq in enumerate(sequences_objs):
        for j, cliff in enumerate(seq):
            tableau_storage[i, j] = cliff.tableau

    # 3. Save
    metaD={
        'rb_seq_len': args.seqLen, 
        'num_rb_seq': args.numSeq,
        'seed': args.seed,
        'rb_type': mode_str,
        'n_qubits': n_qubits
    }
    
    # 4. Verify
    # UPDATED: Pass sequences_circs to verify_sequences for clean visualization
    if args.verify > 0:
        qc = verify_sequences(tableau_storage, sequences_circs, metaD)
    elif args.printCirc:
        qc = verify_sequences(tableau_storage[:1], sequences_circs[:1], metaD)

    if args.printCirc:
        if 'i' in args.printCirc:
            print(qc.draw())
        if 'd' in args.printCirc:
            print('decomposed ideal circuit:'); print(qc.decompose())
        print('ideal gates:', qc.decompose().count_ops())


    bigD = {'rb_tableaus': tableau_storage}
    
    outF = "%s.%s.npz"%(args.dataName,mode_str)
    outFF = os.path.join(args.dataPath, outF)
    
    write_data_npz(bigD, outFF, metaD=metaD, verb=args.verb)
    print(f"  saved: {outFF}")

    print("\n    ./run_one_noisy_RB.py  --rbType %s --inputRB %s    "%(args.rbType,args.dataName))
    
if __name__ == "__main__":
    main()
