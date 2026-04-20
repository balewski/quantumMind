#!/usr/bin/env python3
"""Prepare and run a Bell state on the FakeTorino backend."""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino


def build_bell_circuit():
    """Create a Bell-state circuit with measurements."""
    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def main():
    shots = 4000
    fake_backend = FakeTorino()
    sim_backend = AerSimulator.from_backend(fake_backend)

    qc = build_bell_circuit()
    qc_t = transpile(qc, backend=fake_backend, optimization_level=1, seed_transpiler=123)

    print("Bell circuit:")
    print(qc.draw("text"))
    print(f"Using fake backend: {fake_backend.name}")
    print("\nTranspiled circuit:")
    print(qc_t.draw("text", idle_wires=False))

    job = sim_backend.run(qc_t, shots=shots)
    result = job.result()
    counts = dict(sorted(result.get_counts().items()))

    print(f"\nShots: {shots}")
    print("Counts:", counts)



if __name__ == "__main__":
    main()
