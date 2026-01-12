#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


from qiskit import QuantumCircuit, Aer, execute
from qiskit.providers.aer.noise import NoiseModel, errors
'''
This code was mostly writen by ChatGPT

Clarification:
For a single-qubit gate like the Hadamard ('h'), if we assume the total gate error is evenly distributed across different types of errors, then dividing by 2 was meant to split this error between 'X' and 'Z' errors, assuming these are the primary error types for illustrative purposes. This implies that each type of error (X and Z) could occur with half the total error probability, under a simplistic model where only these errors are considered, and they are equally likely.

For a two-qubit gate like the CNOT ('cx'), the error modeling becomes more complex due to interactions between qubits. A simple division of the error probability might not be appropriate or accurate. In practice, two-qubit gates often exhibit different types and rates of errors, including both local errors on each qubit and correlated errors affecting both qubits simultaneously. The simplified pauli_error used in the example does not fully address these complexities.
'''

#...!...!....................
def ghzCirc(nq=2):
    """Create a GHZ state preparation circuit."""
    qc = QuantumCircuit(nq, nq)
    qc.h(0)
    for i in range(1, nq):
        qc.cx(0, i)
    qc.measure(range(nq), range(nq))
    return qc

#...!...!....................
def set_read_error(cf, noise_model):
    """Configure the noise model based on readout error probabilities."""
    for qubit, probs in enumerate(cf):
        prob_matrix = [[probs['set0meas0'], 1 - probs['set0meas0']],
                       [1 - probs['set1meas1'], probs['set1meas1']]]
        readout_error = errors.ReadoutError(prob_matrix)
        noise_model.add_readout_error(readout_error, [qubit])

#...!...!....................
def set_1q_gate_error(cf, noise_model, qubit):
    """Set single-qubit gate error."""
    T1, T2, duration = cf['T1_us'] * 1e-6, cf['T2_us'] * 1e-6, cf['duration_us'] * 1e-6
    fidelity = cf['fidelity']
    gate_error = 1 - fidelity
    
    pauli_err = errors.pauli_error([('X', gate_error / 2), ('I', 1 - gate_error / 2)])
    thermal_err = errors.thermal_relaxation_error(T1, T2, duration, excited_state_population=0)
    combined_error = pauli_err.compose(thermal_err)
    noise_model.add_quantum_error(combined_error, ['u3'], [qubit])  # 'u3' used as a proxy for any single-qubit gate

def set_2q_gate_error(cf, noise_model, qubits):
    """Set two-qubit gate error."""
    T1, T2, duration = cf['T1_us'] * 1e-6, cf['T2_us'] * 1e-6, cf['duration_us'] * 1e-6
    fidelity = cf['fidelity']
    gate_error = 1 - fidelity
    
    pauli_err = errors.pauli_error([('XX', gate_error / 4), ('YY', gate_error / 4),
                                    ('ZZ', gate_error / 4), ('II', 1 - 3 * gate_error / 4)])
    thermal_err_q1 = errors.thermal_relaxation_error(T1, T2, duration, excited_state_population=0)
    thermal_err_q2 = errors.thermal_relaxation_error(T1, T2, duration, excited_state_population=0)
    combined_error = pauli_err.compose(thermal_err_q1).compose(thermal_err_q2)
    noise_model.add_quantum_error(combined_error, 'cx', qubits)

#=================================
#        M A I N 
#=================================
if __name__ == "__main__":
    nq = 2  # Number of qubits
    # Gate error configuration
    gate_1q_cf = {'T1_us': 110, 'T2_us': 120, 'duration_us': 0.032, 'fidelity': 0.99}
    gate_2q_cf = {'T1_us': 110, 'T2_us': 120, 'duration_us': 0.42, 'fidelity': 0.95}
    # Readout error configuration
    read_cf = [{'set0meas0': 0.91, 'set1meas1': 0.92},
               {'set0meas0': 0.93, 'set1meas1': 0.94}]
    assert nq==len(read_cf)
    
    qc = ghzCirc(nq)
    print(qc)

    #... configure noise model ...
    noise_model = NoiseModel()
    
    for qubit in range(nq): set_1q_gate_error(gate_1q_cf, noise_model, qubit)
    set_2q_gate_error(gate_2q_cf, noise_model, [0, 1])  # Assuming CX between qubits 0 and 1
    set_read_error(read_cf, noise_model)
    
    #.... Execute  circuit with the noise model
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1024, noise_model=noise_model)
    result = job.result()
    counts = result.get_counts(qc)
    
    print("Noisy GHZ Measurement Results:", counts)
