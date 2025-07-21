#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from qiskit import QuantumCircuit,QuantumRegister, transpile, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, random_unitary
from qiskit.quantum_info import Operator

import numpy as np

#...!...!....................
def create_random_unitary_circuit(n,seed=42):
    """Create circuit with structured gates (RY, CX) in linear connectivity"""
    np.random.seed(seed)
    assert n>1, 'n must be greater than 1'
    
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    # Create structured circuit with RY and CX gates
    # Multiple layers for better mixing
    num_layers = 1
    
    for layer in range(num_layers):
        # Layer of RY rotations on all qubits
        for i in range(n):
            theta = np.random.uniform(0, 2*np.pi)
            qc.ry(theta, i)
        
        # Layer of CX gates with linear connectivity
        for i in range(n-1):
            qc.cx(i, i+1)
        
        # Add barrier between layers for clarity
        if layer < num_layers - 1:
            qc.barrier()
    
    print('M: created structured circuit, n_qubits=%d, layers=%d' % (n, num_layers))
    return qc


def compare_unitaries(qc_ideal, qc_transpiled):
    """Compare unitary matrices of ideal and transpiled circuits"""
    print('M: comparing unitaries of circuit pair...')
    print('========\n======\n====')
    print(qc_ideal)
    print(qc_transpiled)
    print('========\n======\n====')
    
    eps = 1e-4

    # Get unitary operators
    U_ideal = Operator(qc_ideal).data
    U_transpiled = Operator(qc_transpiled).data

    # Check dimensions match
    assert U_ideal.shape == U_transpiled.shape, 'dimension mismatch - ideal:%s vs transpiled:%s' % (str(U_ideal.shape), str(U_transpiled.shape))

    # Find element with largest absolute value in ideal unitary
    abs_vals = np.abs(U_ideal)
    i, j = np.unravel_index(np.argmax(abs_vals), abs_vals.shape)
    print('i,j:',i,j,'ideal:',U_ideal[i, j],'transp:',U_transpiled[i, j])
    
    # Compute relative phase between U_ideal(i,j) and U_transpiled(i,j)
    phase0 = U_transpiled[i, j] / U_ideal[i, j]
    print('abs(phase0)=',abs(phase0))

    # Normalize transpiled unitary by phase0
    U_transpiled_normalized = U_transpiled / phase0

    # Element-wise comparison
    diff_matrix = U_ideal - U_transpiled_normalized
    max_real_diff = np.max(np.abs(np.real(diff_matrix)))
    max_imag_diff = np.max(np.abs(np.imag(diff_matrix)))
    max_abs_diff = np.max(np.abs(diff_matrix))

    print('max_elem_index=(%d,%d), phase0=%s' % (i, j, str(phase0)))
    print('max_real_diff=%.6e, max_imag_diff=%.6e, max_abs_diff=%.6e' % (max_real_diff, max_imag_diff, max_abs_diff))

    # Check if differences are within tolerance
    if max_abs_diff > eps:
        print('ERROR: Difference %.6e exceeds tolerance %.6e' % (max_abs_diff, eps))
        return False
    else:
        print('PASS: All differences within tolerance %.6e' % eps)
        return True

def reduce_unused_qubits(qc_transpiled):
    """Remove idle qubits from transpiled circuit to reduce unitary size"""
    # Find qubits that have operations
    used_qubits = []
    for instruction in qc_transpiled.data:
        for qubit in instruction.qubits:
            qubit_idx = qc_transpiled.find_bit(qubit).index
            if qubit_idx not in used_qubits:
                used_qubits.append(qubit_idx)

    print('order of qubits after reduction:',used_qubits)
    if len(used_qubits) < qc_transpiled.num_qubits:
        # Create new circuit with only used qubits
        qc_compact = QuantumCircuit(len(used_qubits))
        
        # Use qubits in the order they appear, don't sort
        qubit_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_qubits)}

        for instruction in qc_transpiled.data:
            new_qubits = [qubit_map[qc_transpiled.find_bit(qubit).index] for qubit in instruction.qubits]
            qc_compact.append(instruction.operation, new_qubits)
        
        print('transpiled qubits=%d, compact qubits=%d' % (qc_transpiled.num_qubits, qc_compact.num_qubits))
        print('used_qubits=%s' % str(used_qubits))
        return qc_compact
    else:
        print('transpiled qubits=%d, no reduction needed' % qc_transpiled.num_qubits)
        return qc_transpiled

# MAIN
def main():
    nq=3
    qc=create_random_unitary_circuit(nq, seed=42)
    print(qc.draw('text', idle_wires=False))

    basis_gates = ['p','sx', 'cz']
    backend1 = AerSimulator()
    qcT=transpile(qc, basis_gates=basis_gates, backend=backend1)
    print('transpiled for',backend1,'basis_gates=',basis_gates)
    print(qcT.draw('text', idle_wires=False))
    compare_unitaries(qc, qcT)


    # qiskit naitive comparioson
    from qiskit.quantum_info import Operator
    # Compare their operators
    are_equiv = Operator(qc).equiv(Operator(qcT))
    print('Qiskit Unitary check1:', are_equiv)


    
    service = QiskitRuntimeService(channel="ibm_quantum_platform",instance='research-eu')
    backends = service.backends()
    print(backends)

    backName='ibm_aachen'
    print('\n repeat on  %s backend ...'%backName)

    backend2 = service.backend(backName)
    print('use backend =', backend2.name )

    qcT2 = transpile(qc, backend=backend2)
    print('transpiled for',backend2)
    print(qcT2.draw('text', idle_wires=False))

    # Compare unitaries at the end
    # Reduce unused qubits once
    qcT3 = reduce_unused_qubits(qcT2)
   
    compare_unitaries(qc, qcT3)
    are_equiv = Operator(qc).equiv(Operator(qcT3))
    print('Qiskit Unitary check2:', are_equiv)


main()
