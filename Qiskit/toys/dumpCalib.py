#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Dump calibration summaries for selected physical qubits.

Args:
  --backendType {1,2,3}
      1 = FakeTorino (medium noise)
      2 = FakeCusco  (high noise)
      3 = real HW via IBM Runtime Service (edit backend name below if needed)
  --physQL <ints...>
      Space-separated physical qubit indices, e.g. "1 3 7"

Output:
  - Per-qubit table for the requested qubits: qubit  sxErr  readErr
  - Per-pair table for requested qubit pairs that have a CZ gate: q-q  CZerr
"""

import argparse
from itertools import combinations
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeCusco, FakeFez, FakeMarrakesh
import qiskit_ibm_runtime.fake_provider as fake_provider
import inspect


def format_val(val, decimals=4, tol=1e-5):
    if val is None:
        return 'n/a '
    if abs(val) < tol:
        return '0.0'
    v = round(float(val), decimals)
    return f'{v:.{decimals}f}'


def get_backend(backend_type: int):
    if backend_type == 1:
        #backend = FakeTorino(); print('Backend: FakeTorino')
        #backend = FakeMarrakesh (); print('Backend: FakeMarrakesh')
        backend = FakeFez (); print('\nBackend: FakeFez')
    elif backend_type == 2:
        backend = FakeCusco()
        print('\nBackend: FakeCusco')
    elif backend_type == 3:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        # Choose a real backend; adjust as needed
        
        backName = 'ibm_fez'
        #backName = 'ibm_pittsburgh'
        #backName = 'ibm_kingston'
        #backName = 'ibm_marrakesh'
        
        print('\n real HW   %s backend ...' % backName)
        backend = service.backend(backName)
    else:
        raise ValueError('backendType must be one of 1,2,3')
    return backend


def list_available_fake_backends():
    names = []
    for name, obj in vars(fake_provider).items():
        if name.startswith('Fake') and inspect.isclass(obj):
            names.append(name)
    names.sort()
    print('\nAvailable fake backends in qiskit_ibm_runtime.fake_provider:')
    if names:
        print(' '.join(names))
    else:
        print('none found')


def extract_qubit_readout_err(qprops):
    # Try standard fields first, else derive from prob_measX_prepY
    val = None
    p01 = None
    p10 = None
    for item in qprops:
        name = getattr(item, 'name', '')
        if name == 'readout_error':
            val = getattr(item, 'value', None)
        elif name == 'prob_meas1_prep0':
            p01 = getattr(item, 'value', None)
        elif name == 'prob_meas0_prep1':
            p10 = getattr(item, 'value', None)
    if val is None and (p01 is not None and p10 is not None):
        try:
            val = 0.5 * (float(p01) + float(p10))
        except Exception:
            val = None
    return val


def extract_gate_error(gate_props):
    if gate_props is None:
        return None
    for p in gate_props:
        if getattr(p, 'name', '') == 'gate_error':
            return getattr(p, 'value', None)
    return None


def collect_per_qubit(properties, phys_ql):
    sx_err = {q: None for q in phys_ql}
    read_err = {q: None for q in phys_ql}
    t1_map = {q: None for q in phys_ql}
    t2_map = {q: None for q in phys_ql}

    # Readout errors from qubit properties
    for q in phys_ql:
        if q < len(properties.qubits):
            qprops = properties.qubits[q]
            read_err[q] = extract_qubit_readout_err(qprops)
            # Extract T1/T2 (case-insensitive)
            for item in qprops:
                name = getattr(item, 'name', '')
                val = getattr(item, 'value', None)
                if not isinstance(name, str):
                    continue
                lname = name.lower()
                if lname.startswith('t1'):
                    t1_map[q] = val
                elif lname.startswith('t2'):
                    t2_map[q] = val

    # Gate errors: prefer 'sx', else fall back to 'x'
    for gate in properties.gates:
        name = getattr(gate, 'gate', '')
        qubits = getattr(gate, 'qubits', [])
        if len(qubits) == 1 and qubits[0] in phys_ql and name in ('sx', 'x'):
            val = extract_gate_error(getattr(gate, 'parameters', None))
            # Prefer sx; only overwrite x if sx wasn't set
            if name == 'sx' or sx_err[qubits[0]] is None:
                sx_err[qubits[0]] = val

    return sx_err, read_err, t1_map, t2_map


def collect_cz_pairs(properties, phys_ql):
    # Keep only one orientation per pair: use sorted tuple as key
    cz_map = {}  # (minQ, maxQ) -> err
    phys_set = set(phys_ql)
    for gate in properties.gates:
        name = getattr(gate, 'gate', '')
        qubits = getattr(gate, 'qubits', [])
        if name == 'cz' and len(qubits) == 2:
            q0, q1 = qubits
            if q0 in phys_set and q1 in phys_set:
                key = tuple(sorted((q0, q1)))
                if key not in cz_map:  # drop reversed duplicates
                    err = extract_gate_error(getattr(gate, 'parameters', None))
                    cz_map[key] = err
    # Convert to sorted list of (q0, q1, err)
    cz_errs = [(a, b, cz_map[(a, b)]) for a, b in sorted(cz_map.keys())]
    return cz_errs


def print_tables(phys_ql, sx_err, read_err, cz_errs, t1_map, t2_map):
    # Per-qubit table
    print('\nPer-qubit calibration:')
    print('qubit   sxErr   readErr   T1(us)    T2(us)')
    for q in phys_ql:
        def fmt_f1(v):
            try:
                return f'{float(v):6.1f}'
            except Exception:
                return '  n/a '
        print(f'{q:5d}  {fmt_f1(sx_err[q])}   {fmt_f1(read_err[q])}   {fmt_f1(t1_map[q])}    {fmt_f1(t2_map[q])}')

    # CZ pairs
    print('\nCZ connectors among selected qubits:')
    print('q-q    CZerr')
    if not cz_errs:
        print('none')
    else:
        for q0, q1, err in sorted(cz_errs):
            print(f'{q0}-{q1:>2d}  {format_val(err)}')


def format_calib_date_pt(properties):
    if properties is None:
        return 'n/a'
    dt = getattr(properties, 'last_update_date', None)
    if isinstance(dt, list):
        # choose the latest available datetime
        dt = max((d for d in dt if isinstance(d, datetime)), default=None)
    if dt is None:
        return 'n/a'
    if dt.tzinfo is None:
        try:
            dt = dt.replace(tzinfo=ZoneInfo('UTC'))
        except Exception:
            return 'n/a'
    try:
        dt_pt = dt.astimezone(ZoneInfo('America/Los_Angeles'))
        now_pt = datetime.now(ZoneInfo('America/Los_Angeles'))
        mins = int(max(0, round((now_pt - dt_pt).total_seconds() / 60.0)))
        return dt_pt.strftime('%Y-%m-%d %H:%M:%S %Z') + f'  ({mins} min ago)'
    except Exception:
        return 'n/a'


def get_parser():
    parser = argparse.ArgumentParser(
        description='Dump calibration for selected physical qubits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-b', '--backendType', type=int, default=1, choices=[1, 2, 3],
                        help='Backend type: 1=FakeTorino, 2=FakeCusco, 3=real HW')
    parser.add_argument('--physQL', type=int, nargs='+', default=[26, 27, 28, 29],
                        help='Space-separated physical qubit list (e.g., 1 3 7)')
    return parser


def main(args):
    list_available_fake_backends()
    backend = get_backend(args.backendType)
    phys_ql = args.physQL
    print(f'Qubits of interest: {phys_ql}')

    properties = backend.properties()

    print(f'Calibration date (PT): {format_calib_date_pt(properties)}')

    sx_err, read_err, t1_map, t2_map = collect_per_qubit(properties, phys_ql)
    cz_errs = collect_cz_pairs(properties, phys_ql)

    print_tables(phys_ql, sx_err, read_err, cz_errs, t1_map, t2_map)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)


