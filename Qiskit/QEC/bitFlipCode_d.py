#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
Code Summary
This script is a Qiskit simulation of the d-qubit repetition code (default d=5),
a quantum error-correcting code designed specifically to protect against bit-flip (X) errors.
The program can encode a logical state of |0>, |1>, or a superposition into d physical qubits,
apply optional transversal gates, inject up to (d-1)/2 user-defined bit-flip errors,
and then successfully detect them.
It uses a syndrome measurement technique to identify the error locations without disturbing the
logical information and verifies that the decoder correctly identified the injected errors.

Command-Line Arguments
-x / --xq [Qubit_Number(s)]: Specifies which qubits to apply a bit-flip (X) error to.
-d / --dist [int]: Code distance (odd integer, default 5).
-g / --gates [string]: Apply transversal gates before error injection. String containing chars 'x', 's'.
'''

import argparse
import numpy as np
from itertools import combinations
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from pprint import pprint

class BitFlipCode:
    def __init__(self, args):
        self.args = args
        self.d = args.dist
        assert self.d % 2 == 1, "Distance must be an odd integer"
        self.t = (self.d - 1) // 2  # number of correctable errors
        assert self.t >= 1, "Distance must be at least 3"

        self.data_qubits = QuantumRegister(self.d, name='q')
        self.anc_qubits = QuantumRegister(self.d - 1, name='a')
        self.synd_bits = ClassicalRegister(self.d - 1, name='s')
        self.qc = QuantumCircuit(self.data_qubits, self.anc_qubits, self.synd_bits)
        
        self.decoder_map = self.build_decoder_map()
        self.print_decoder_map()

    def build_decoder_map(self):
        """Builds the mapping from a syndrome string to the required correction."""
        d = self.d
        t = self.t
        decoder_map = {}
        
        # Precompute syndrome per qubit
        syndrome_per_qubit = {}
        for q in range(d):
            synd = 0
            if q > 0: synd |= (1 << (q - 1))
            if q < d - 1: synd |= (1 << q)
            syndrome_per_qubit[q] = synd

        num_synd_bits = d - 1
        # 0 errors
        decoder_map[f'{0:0{num_synd_bits}b}'] = ('No error', None)
        
        # 1 to t errors
        for k in range(1, t + 1):
            for locs in combinations(range(d), k):
                combined_synd = 0
                for q in locs:
                    combined_synd ^= syndrome_per_qubit[q]
                synd_str = f'{combined_synd:0{num_synd_bits}b}'
                val = locs[0] if k == 1 else list(locs)
                decoder_map[synd_str] = ('X', val)
                
        return decoder_map

    def print_decoder_map(self):
        print(f'Mapping from classical syndrome to correction (dist={self.d}):')
        if self.d > 5:
            print(f"  (Printing sample of {min(5, len(self.decoder_map))} entries due to large size: {len(self.decoder_map)} entries)")
            pprint(dict(list(self.decoder_map.items())[:5]))
        else:
            print(' syndrom:  action'); pprint(self.decoder_map)

    def get_syndrome_circuit(self):
        """Returns a circuit for measuring the d-1 stabilizers: Z0Z1, ... Z(d-2)Z(d-1)."""
        d = self.d
        q = QuantumRegister(d, name='q')
        a = QuantumRegister(d-1, name='a')
        qc = QuantumCircuit(q, a, name='Syndrome')
        for i in range(d-1):
            qc.cx(q[i], a[i])
            qc.cx(q[i+1], a[i])
        qc.barrier()
        return qc

    def init_0L(self):
        """Initializes the state of the data qubits based on arguments."""    
        print("--- Initializing to logical state |0_L> ---")
        # does nothing for btFlip code (state is |00...0>)

    def transversalU(self, gate_char):
        """Applies transversal gates X or S to all data qubits (H removed)."""
        if gate_char.lower() == 'x':
            self.qc.x(self.data_qubits)
        elif gate_char.lower() == 's':
            self.qc.s(self.data_qubits)
        else:
            print(f"Warning: Unknown or unsupported transversal gate '{gate_char}', skipping.")

    def inject_errors(self):
        x_error_qubits = self.args.xq
        if x_error_qubits:
            print(f"--- Injecting X error(s) on qubit(s): {x_error_qubits} ---")
            for q in x_error_qubits:
                if 0 <= q < self.d:
                    self.qc.x(q)
                else:
                    print(f"Warning: Qubit {q} out of range (0-{self.d-1}), skipping.")
            self.qc.barrier()

    def build_circuit(self):
        # Step 1: Prepare and Encode
        self.init_0L()

        # Step 1b: Apply Transversal Gates
        if self.args.gates:
            print(f"--- Applying transversal gates: {self.args.gates} ---")
            for g in self.args.gates:
                self.transversalU(g)
            self.qc.barrier()

        # Step 2: Inject errors
        self.inject_errors()

        # Step 3: Measure syndromes
        self.qc.append(self.get_syndrome_circuit().to_instruction(), self.qc.qubits)
        self.qc.measure(self.anc_qubits, self.synd_bits)
        if self.args.verb <= 1: print(self.qc)
        else: print(self.qc.decompose())
        return self.qc

    def evaluate(self, measured_syndrome):
        correction = self.decoder_map.get(measured_syndrome)
        gate, detected_qubits = (None, None) if not correction else correction
        
        if not correction:
            print("Error: Syndrome not found in decoder map! Cannot correct.")
        else:
            print(f"Decoded Correction: Apply '{gate}' gate to qubit(s) {detected_qubits}")

        # --- 5. Decoder Verification ---
        print("\n--- Verifying Decoder Correctness ---")
        
        # Standardize the injected error locations for comparison by sorting them.
        injected_errors_sorted = sorted(self.args.xq)
        
        # Standardize the decoded error locations.
        decoded_errors_sorted = []
        if detected_qubits is not None:
            if isinstance(detected_qubits, list):
                decoded_errors_sorted = sorted(detected_qubits)
            else: # It's a single integer for a one-qubit error
                decoded_errors_sorted = [detected_qubits]

        print(f"Injected error location(s): {injected_errors_sorted if injected_errors_sorted else 'None'}")
        print(f"Decoded error location(s):  {decoded_errors_sorted if decoded_errors_sorted else 'None'}")
        
        # Compare the standardized lists.
        isSuccess = injected_errors_sorted == decoded_errors_sorted
        if isSuccess:
            print("\n*** DECODER VERIFICATION PASSED ***\n")
        else:
            print("\n*** DECODER VERIFICATION FAILED ***\n")
        return isSuccess

def main():
    parser = argparse.ArgumentParser(description='Repetition Code Simulation (Bit-Flip Protection)')
    parser.add_argument('-d', '--dist', type=int, default=5, help='Code distance (odd integer, default 5)')
    parser.add_argument('-x', '--xq', type=int, nargs='+', default=[1], help='List of qubits to apply X error on (e.g., -x 1 3)')
    parser.add_argument('-g', '--gates', type=str, default='', help='Apply transversal gates (x,s) before errors')
    parser.add_argument("-v", "--verb", type=int, help="Increase debug verbosity", default=1)
    args = parser.parse_args()
    
    codeObj = BitFlipCode(args)
    qc = codeObj.build_circuit()
    print('M:  num_qubits:=%d, dist=%d, syndr_bits:=%d\n gates=%s'%(qc.num_qubits, args.dist, qc.num_clbits, qc.count_ops()))
    
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    qcT = pm.run(qc)
    result = backend.run(qcT, shots=1).result()
    counts = result.get_counts()
    meas_synd = list(counts.keys())[0]
    print(f"\nMeasured Syndrome: {meas_synd}")

    codeObj.evaluate(meas_synd)

if __name__ == "__main__":
    main()
