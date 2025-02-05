#!/usr/bin/env python3

'''
Applying noise to custom unitary gates

https://qiskit.github.io/qiskit-aer/tutorials/4_custom_gate_noise.html
'''

from qiskit import transpile, QuantumCircuit
import qiskit.quantum_info as qi  # IMPORTANT for defining custom gates

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error,  depolarizing_error, thermal_relaxation_error, ReadoutError

move_op = qi.Operator([[1, 0],
                        [0, 1]])

#.... Using operators in circuits

qc = QuantumCircuit(3, name='abc')

# Add gates
qc.h(0)
qc.cx(0,1)
qc.cx(0,2)
qc.unitary(move_op, [2], label='move1')
qc.unitary(move_op, [2], label='move1')
qc.measure_all()

print('ideal circuit:')
print(qc)

assert 'unitary' in AerSimulator().configuration().basis_gates

# qubit property???
t1 = 50e-6  # 50 us
t2 = 70e-6  # 70 us

# Error parameters
gate_time=10.e-9  # us
p_depol2 = 0.03   # Two-qubit depolarizing error probability

# Construct the error
thermal_error = thermal_relaxation_error(t1, t2, gate_time)
depol_error = depolarizing_error(p_depol2, 1)
combined_error = depol_error.compose(thermal_error)

# Build the noise model by adding the error to the move-gate
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(combined_error, 'move1')

print('\n Check  the desired gate string is in the noise model basis_gates ')
noise_model.add_basis_gates(['unitary'])
print(noise_model.basis_gates)


print('\n\n Simulating a custom unitary noise model')

print('\n ideal simu')
shots=2000
sim_ideal = AerSimulator()
ideal_result = sim_ideal.run(qc,shots=shots).result()
ideal_counts = ideal_result.get_counts(0)
print('ideal_counts',ideal_counts)


print('\n noisy simu')
sim_noise = AerSimulator(noise_model=noise_model)

# Run on the simulator without noise
noise_result = sim_noise.run(qc,shots=shots).result()
noise_counts = noise_result.get_counts(0)
print('noise_counts (MSBF):',noise_counts)


print('transpilation will preserve unitary for level=1')
qcT2 = transpile(qc, sim_noise,optimization_level=0)
print(qcT2)
