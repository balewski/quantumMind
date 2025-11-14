#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Standard CNOT truth tables

Computes two 4x4 truth tables for stacked CNOTs acting on a fixed
control-target pair of qubits:
- Z basis: inputs 00, 01, 10, 11 with computational-basis measurements
- X basis: inputs ++, +-, -+, -- with X-basis measurements

Usage:
  ./doCNOT_truthTable.py --nshot 50000 --backendType 0
  ./doCNOT_truthTable.py --stack 2 --nshot 10000

Backend types:
  0 = AerSimulator (ideal, perfect)
  1 = FakeTorino (medium noise)
  2 = FakeCusco (high noise)
  3 = ibm_fez  - real

Register naming:
  qctr: control qubit
  qtrg: target qubit
  mct:  classical register collecting final measurements
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.circuit import MidCircuitMeasure
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeCusco, FakeFez, FakeMarrakesh
import argparse
import time


def format_prob(prob, decimals=3, tol=1e-5):
    """Format probabilities with thresholding and fixed decimal precision."""
    if abs(prob) < tol:
        return '0.000'

    rounded = round(prob, decimals)
    return f'{rounded:.{decimals}f}'


def create_standard_cnot_circuit(input_state, basis='Z', n_stack=1):
    """Create a circuit implementing stacked CNOTs for a given input state."""
    qctr = QuantumRegister(1, 'qctr')
    qtrg = QuantumRegister(1, 'qtrg')
    mct = ClassicalRegister(2, 'mct')
    qc = QuantumCircuit(qctr, qtrg, mct)

    if basis == 'Z':
        if input_state[0] == '1':
            qc.x(qctr[0])
        if input_state[1] == '1':
            qc.x(qtrg[0])
    elif basis == 'X':
        if input_state[0] == '+':
            qc.h(qctr[0])
        elif input_state[0] == '-':
            qc.x(qctr[0])
            qc.h(qctr[0])
        else:
            raise ValueError(f'Unknown X-basis symbol for control: {input_state[0]}')

        if input_state[1] == '+':
            qc.h(qtrg[0])
        elif input_state[1] == '-':
            qc.x(qtrg[0])
            qc.h(qtrg[0])
        else:
            raise ValueError(f'Unknown X-basis symbol for target: {input_state[1]}')
    else:
        raise ValueError(f'Unsupported basis: {basis}')

    qc.barrier()
    for _ in range(n_stack):
        qc.cx(qctr[0], qtrg[0])
    qc.barrier()

    if basis == 'X':
        qc.h(qctr[0])
        qc.h(qtrg[0])

    qc.measure(qctr[0], mct[0])
    qc.measure(qtrg[0], mct[1])
    # attach metadata for easy selection, format like '01z' or '11x'
    if basis == 'Z':
        bits = input_state
        basis_char = 'z'
    else:
        b0 = '0' if input_state[0] == '+' else '1'
        b1 = '0' if input_state[1] == '+' else '1'
        bits = b0 + b1
        basis_char = 'x'
    qc.metadata = qc.metadata or {}
    qc.metadata['type'] = bits + basis_char
    return qc


def compute_ideal_truth_tables(n_stack=1):
    """Return hardcoded ideal truth tables for Z and X bases.

    - For odd n_stack: single CNOT behavior
      Z-basis mapping: (c,t) -> (c, c^t)
      X-basis mapping: (c,t) -> (c^t, t)  [CNOT with roles swapped under X]
    - For even n_stack: Identity in both bases
    """
    basis_Z_inputs = ['00', '01', '10', '11']
    basis_X_inputs = ['++', '+-', '-+', '--']
    basis_Z_outputs = basis_Z_inputs
    basis_X_outputs = basis_X_inputs

    truth_Z = np.zeros((4, 4))
    truth_X = np.zeros((4, 4))

    if n_stack % 2 == 0:
        # Identity for both bases
        truth_Z = np.eye(4)
        truth_X = np.eye(4)
    else:
        # Z basis: CNOT mapping (c,t)->(c, c^t)
        z_map = {
            0: 0,  # 00 -> 00
            1: 1,  # 01 -> 01
            2: 3,  # 10 -> 11
            3: 2,  # 11 -> 10
        }
        for r in range(4):
            truth_Z[r, z_map[r]] = 1.0

        # X basis: swapped-CNOT mapping (c,t)->(c^t, t)
        x_map = {
            0: 0,  # 00 -> 00  (++,++)
            1: 3,  # 01 -> 11  (+-,--)
            2: 2,  # 10 -> 10  (-+, -+)
            3: 1,  # 11 -> 01  (--,+-)
        }
        for r in range(4):
            truth_X[r, x_map[r]] = 1.0

    return {
        'Z': (truth_Z, basis_Z_inputs, basis_Z_outputs),
        'X': (truth_X, basis_X_inputs, basis_X_outputs)
    }


def default_entGate(qc: QuantumCircuit, ctr, trg):
    """Default entangling gate: CNOT(ctr->trg)."""
    qc.cx(ctr, trg)
    qc.barrier()


def teleCnot(qc: QuantumCircuit, ctr, trg):
    """Teleported CNOT using per-layer ancilla pair and classical corrections.

    Expects ancilla quantum registers named anc{i} and classical registers ab{i}
    to exist on the circuit for each stacked layer. Uses qc.metadata['num_stack']
    to choose the next layer index and increments it after use.
    """
    layer_idx = qc.metadata.get('num_stack', 0)

    # Locate ancilla and classical registers for this layer
    anc = next(r for r in qc.qregs if r.name == f'anc{layer_idx}')
    creg = next(r for r in qc.cregs if r.name == f'ab{layer_idx}')

    # Create entangled pair between processors
    qc.h(anc[0])
    qc.cx(anc[0], anc[1])
    qc.barrier()

    # Teleport CNOT gate
    qc.cx(ctr, anc[0])
    qc.cx(anc[1], trg)

    qc.h(anc[1])
    # Mid-circuit measurements using MidCircuitMeasure
    mid_instr = MidCircuitMeasure()
    qc.append(mid_instr, [anc[0]], [creg[0]])
    qc.append(mid_instr, [anc[1]], [creg[1]])
    # Conditional corrections using c_if with measured bits
    qc.x(trg).if_test(creg[0], 1)  # X correction on target
    qc.z(ctr).c_if(creg[1], 1)  # Z correction on control
    qc.barrier()

    # Advance layer pointer
    qc.metadata['num_stack'] = layer_idx + 1


def build_all_circuits(n_stack=1, entGate=default_entGate, teleCNOT=False):
    """Construct and return the list of 8 circuits in a fixed order using entGate.

    The function builds circuits with proper input state preparation and
    measurement basis. The only injected operation is entGate(qc, ctr, trg)
    applied n_stack times.

    Order: Z-basis [00, 01, 10, 11] then X-basis [++, +-, -+, --].
    """
    circuits = []

    # Create a single blank template circuit with named registers
    qctr_reg = QuantumRegister(1, 'qctr')
    qtrg_reg = QuantumRegister(1, 'qtrg')
    mct_reg = ClassicalRegister(2, 'mct')

    if teleCNOT:
        # Add ancilla and classical regs for each stacked layer
        anc_regs = [QuantumRegister(2, f'anc{i}') for i in range(n_stack)]
        ab_regs = [ClassicalRegister(2, f'ab{i}') for i in range(n_stack)]
        template = QuantumCircuit(qctr_reg, qtrg_reg, *anc_regs, mct_reg, *ab_regs)
    else:
        template = QuantumCircuit(qctr_reg, qtrg_reg, mct_reg)

    # Z-basis inputs
    for state in ['00', '01', '10', '11']:
        qc = template.copy()
        qc.metadata = {}
        if teleCNOT:
            qc.metadata['num_stack'] = 0

        # Resolve registers from the copied circuit
        qctr = next(r for r in qc.qregs if r.name == 'qctr')
        qtrg = next(r for r in qc.qregs if r.name == 'qtrg')
        mct = next(r for r in qc.cregs if r.name == 'mct')

        # Prepare computational-basis input
        if state[0] == '1':
            qc.x(qctr[0])
        if state[1] == '1':
            qc.x(qtrg[0])
        qc.barrier()

        # Apply stacked entangling operation
        for _ in range(n_stack):
            entGate(qc, qctr[0], qtrg[0])

        # Z-basis measurement
        qc.measure(qctr[0], mct[0])
        qc.measure(qtrg[0], mct[1])

        # Attach metadata type like '01z'
        qc.metadata['type'] = state + 'z'
        circuits.append(qc)

    # X-basis inputs
    for label in ['++', '+-', '-+', '--']:
        qc = template.copy()
        qc.metadata = {}
        if teleCNOT:
            qc.metadata['num_stack'] = 0

        qctr = next(r for r in qc.qregs if r.name == 'qctr')
        qtrg = next(r for r in qc.qregs if r.name == 'qtrg')
        mct = next(r for r in qc.cregs if r.name == 'mct')

        # Prepare X-basis input
        if label[0] == '+':
            qc.h(qctr[0])
        else:  # '-'
            qc.x(qctr[0]); qc.h(qctr[0])
        if label[1] == '+':
            qc.h(qtrg[0])
        else:  # '-'
            qc.x(qtrg[0]); qc.h(qtrg[0])
        qc.barrier()
        
        # Apply stacked entangling operation
        for _ in range(n_stack):
            entGate(qc, qctr[0], qtrg[0])
    

        # Rotate back for X-basis measurement
        qc.h(qctr[0]); qc.h(qtrg[0])
        qc.measure(qctr[0], mct[0])
        qc.measure(qtrg[0], mct[1])

        # Attach metadata type like '11x' mapping +/- -> 0/1
        b0 = '0' if label[0] == '+' else '1'
        b1 = '0' if label[1] == '+' else '1'
        qc.metadata['type'] = b0 + b1 + 'x'
        circuits.append(qc)

    print(f'Created {len(circuits)} circuits for truth tables (stack={n_stack})')
    return circuits


def evaluate_and_print_results(result, ideal_tables, nshot, backendName, n_stack=1):
    """Evaluate sampler results against ideal tables and print formatted rows with discrepancy."""
    basis_order = [('Z', ['00', '01', '10', '11'], ['00', '01', '10', '11']),
                   ('X', ['++', '+-', '-+', '--'], ['++', '+-', '-+', '--'])]

    idx = 0
    print(f'\nMeasured truth tables (shots={nshot}), backend={backendName}:')
    print('Input -> Output  probabilities [± stat error]               |  L1(all) |  err@1')

    for basis, inputs, outputs in basis_order:
        truth_ideal, _, _ = ideal_tables[basis]
        print(f'\nBasis {basis}:')
        for row, inp_label in enumerate(inputs):
            pub_result = result[idx]
            counts = pub_result.data.mct.get_counts()
            idx += 1

            total_shots = sum(counts.values())
            measured_row = np.zeros(4)
            if total_shots > 0:
                for bitstring, count in counts.items():
                    bits = bitstring.split()[0] if isinstance(bitstring, str) and ' ' in bitstring else bitstring
                    ctrl_bit = int(bits[1])
                    targ_bit = int(bits[0])
                    if basis == 'Z':
                        out_label = f'{ctrl_bit}{targ_bit}'
                    else:
                        ctrl_sign = '+' if ctrl_bit == 0 else '-'
                        targ_sign = '+' if targ_bit == 0 else '-'
                        out_label = ctrl_sign + targ_sign
                    col = outputs.index(out_label)
                    measured_row[col] += count / total_shots

            probs_str = []
            for j in range(4):
                prob = measured_row[j]
                if prob > 0 and prob < 1:
                    std_err = np.sqrt(prob * (1 - prob) / nshot)
                else:
                    std_err = 1.0 / nshot
                prob_str = format_prob(prob)
                err_str = format_prob(std_err)
                probs_str.append(f'{prob_str}±{err_str}')

            discrepancy = np.sum(np.abs(measured_row - truth_ideal[row]))
            disc_str = format_prob(discrepancy)
            j_true = int(np.argmax(truth_ideal[row]))
            err_at_one = abs(1.0 - measured_row[j_true])
            err1_str = format_prob(err_at_one)
            print(f'  {inp_label}: [{", ".join(probs_str)}]  |  {disc_str} |  {err1_str}')

    return


def print_selected_circuit(circuits, selector):
    """Print the circuit whose input and basis match the selector like '01z' or '11x'.

    For Z basis: selector first two chars should be 0/1 and last char 'z'.
    For X basis: selector first two chars 0/1 are mapped to +/- respectively, last char 'x'.
    """
    if selector is None or len(selector) != 3:
        return

    desired = selector.lower()
    for qc in circuits:
        meta = getattr(qc, 'metadata', None)
        if isinstance(meta, dict) and meta.get('type', '').lower() == desired:
            print('\n' + str(qc.metadata) + '\n' + qc.draw(output='text', fold=-1, idle_wires=False).__str__())
            print()
            return




def get_parser():
    parser = argparse.ArgumentParser(
        description='Stacked CNOT truth tables in Z and X bases',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-n', '--nshot', type=int, default=10_000,
                        help='Number of shots for circuit execution')
    parser.add_argument('-b', '--backendType', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Backend type: 0=ideal (AerSimulator), 1=FakeTorino, 2=FakeCusco, 3=HW')
    parser.add_argument('-s', '--stack', type=int, default=1,
                        help='Number of CNOT gates to stack (CNOT^n)')
    parser.add_argument('--physQL', type=int, nargs='+', default=None,
                        help='Space-separated physical qubit list for initial_layout (e.g., 1 3)')
    parser.add_argument('--printCirc', type=str, default=None,
                        help='Print circuit matching selector like 01z or 11x (two bits + basis)')
    parser.add_argument('--transSeed', type=int, default=42,
                        help='Seed for the transpiler randomness')
    parser.add_argument('--teleCNOT', action='store_true',
                        help='Use teleported CNOT as the entangling gate')
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")

    return parser


def get_backend(backend_type: int):
    if backend_type == 0:
        backend = AerSimulator()
        print('Backend: AerSimulator (ideal)')
    elif backend_type == 1:
        #backend = FakeTorino(); print('Backend: FakeTorino')
        backend = FakeMarrakesh (); print('Backend: FakeMarrakesh')
        backend = FakeFez (); print('Backend: FakeFez')
    elif backend_type == 2:
        backend = FakeCusco()
        print('Backend: FakeCusco')
    elif backend_type == 3:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backName = 'ibm_fez'
        #backName = 'ibm_pittsburgh'
        #backName = 'ibm_kingston'
        #backName = 'ibm_marrakesh'
        print('\n real HW   %s backend ...' % backName)
        backend = service.backend(backName)
    else:
        raise ValueError(f'Unsupported backendType: {backend_type}')
    return backend


def main(args):
    nshot = args.nshot
    n_stack = args.stack

    backend = get_backend(args.backendType)

    print(f'Stacking {n_stack} standard CNOT gate(s)')

    print(f'\nIdeal truth tables for CNOT^{n_stack}...')
    ideal_tables = compute_ideal_truth_tables(n_stack)

    for basis in ['Z', 'X']:
        truth, inputs, outputs = ideal_tables[basis]
        print(f'\nIdeal truth table ({basis}-basis):')
        for i, inp in enumerate(inputs):
            formatted_row = [format_prob(val) for val in truth[i]]
            print(f'  {inp}: [{", ".join(formatted_row)}]')

    print(f'\nConstruct 8 circuits for measured truth tables...')
    ent_gate = teleCnot if args.teleCNOT else default_entGate
    qcL  = build_all_circuits(n_stack, ent_gate, teleCNOT=args.teleCNOT)
    if args.printCirc is not None:
        print_selected_circuit(qcL, args.printCirc)

    for i, qc in enumerate(qcL):
        len2q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
        print(f'c{i} meta:{qc.metadata}  len2q={len2q}, {qc.count_ops()}')
    phys_layout = args.physQL if args.physQL is not None else None
    print(f'\nTranspile 8 circuits for backend {backend.name}...')
    
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=args.transSeed, initial_layout=phys_layout)
    qcTL = pm.run(qcL, num_processes=1)
    print(f'Transpiled circuits for {backend.name}, seed={args.transSeed}')
    
    for i, qc in enumerate(qcTL):
        len2q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
        if args.backendType>0:
            physQubitLayout = qc._layout.final_index_layout(filter_ancillas=True)
        else:
            physQubitLayout=[]
        nqTot = len(physQubitLayout)
        print(f'qc{i} meta:{qc.metadata}  2Q={len2q}  physQL={physQubitLayout}  nqTot={nqTot}')
    if args.printCirc is not None:
        print_selected_circuit(qcTL, args.printCirc)
    if not args.executeCircuit:
        print('\nNO execution of circuit, use -E to execute the job\n')
        exit(0)
  
    print(f'\nRun 8 circuits  on {backend.name}...')
    sampler = Sampler(mode=backend)
    t0 = time.time()
    job = sampler.run(qcTL, shots=nshot)
    result = job.result()
    elapsed = time.time() - t0
    print(f'execution completed in {elapsed:.2f} seconds')

    evaluate_and_print_results(result, ideal_tables, nshot, backend.name, n_stack)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    for arg in vars(args): print( 'myArgs:',arg, getattr(args, arg))
    main(args)


