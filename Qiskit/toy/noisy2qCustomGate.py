#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Applying noise to custom unitary gates

https://qiskit.github.io/qiskit-aer/tutorials/4_custom_gate_noise.html
'''

from qiskit import transpile, QuantumCircuit
import qiskit.quantum_info as qi  # IMPORTANT for defining custom gates

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error

from qiskit.visualization import plot_histogram


# CNOT matrix operator with qubit-0 as control and qubit-1 as target
cx_op = qi.Operator([[1, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0]])

# iSWAP matrix operator
iswap_op = qi.Operator([[1, 0, 0, 0],
                        [0, 0, 1j, 0],
                        [0, 1j, 0, 0],
                        [0, 0, 0, 1]])

# Note: The matrix is specified with respect to the tensor product  Ub(x)Ua for qubits specified by list [a, b].


#.... Using operators in circuits

# CNOT in terms of iSWAP and single-qubit gates
cx_circ = QuantumCircuit(2, name='cx<iSWAP>')

# Add gates
cx_circ.sdg(1)
cx_circ.h(1)
cx_circ.sdg(0)
cx_circ.unitary(iswap_op, [0, 1], label='iswap')
cx_circ.sdg(0)
cx_circ.h(0)
cx_circ.sdg(0)
cx_circ.unitary(iswap_op, [0, 1], label='iswap')
cx_circ.s(1)

print(cx_circ)

print('\n confirm it is unitary:')
# Simulate the unitary for the circuit using Operator:
unitary = qi.Operator(cx_circ)
print(unitary)


print('\ncompute fidelity of whole circuit , ideal ??')
f_ave = qi.average_gate_fidelity(cx_op, unitary)
print("Average Gate Fidelity: F = {:f}".format(f_ave))


print('\n check if Qiskit Aer AerSimulator supports simulation of arbitrary unitary operators directly:')
print('unitary' in AerSimulator().configuration().basis_gates)


# This allows us to add noise models to arbitrary unitaries in our simulation when we identify them using the optional label argument of QuantumCircuit.unitary.

# Error parameters
param_q0 = 0.05  # damping parameter for qubit-0
param_q1 = 0.1   # damping parameter for qubit-1

# Construct the error
qerror_q0 = amplitude_damping_error(param_q0)
qerror_q1 = amplitude_damping_error(param_q1)
iswap_error = qerror_q1.tensor(qerror_q0)

# Build the noise model by adding the error to the "iswap" gate
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(iswap_error, 'iswap')

print('\n Check  the desired gate string is in the noise model basis_gates ')
noise_model.add_basis_gates(['unitary'])
print(noise_model.basis_gates)


print('\n\n Simulating a custom unitary noise model')
# Bell state circuit where iSWAPS should be inserted at barrier locations
bell_circ = QuantumCircuit(2, 2, name='bell')
bell_circ.h(0)
bell_circ.append(cx_circ, [0, 1])
bell_circ.measure([0,1], [0,1])
print(bell_circ)

print('\n ideal simu')
sim_ideal = AerSimulator()
tbell_circ = transpile(bell_circ, sim_ideal)

ideal_result = sim_ideal.run(tbell_circ).result()
ideal_counts = ideal_result.get_counts(0)
print('ideal_counts',ideal_counts)


print('\n noisey simu')
sim_noise = AerSimulator(noise_model=noise_model)
tbell_circ_noise = transpile(bell_circ, sim_noise)

# Run on the simulator without noise
noise_result = sim_noise.run(tbell_circ_noise).result()
noise_counts = noise_result.get_counts(bell_circ)
print('noise_counts:',noise_counts)
