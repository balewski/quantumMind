#!/usr/bin/env python3
'''
SUMMARY: Distance-3 Repetition Code for Bit-Flip Errors
========================================================

Circuit Structure:
- 5 qubits total: 
  * Data qubits: 0, 2, 4 (store the logical qubit)
  * Ancilla qubits: 1, 3 (measure error syndromes)

- Qubit 1 detects errors between qubits 0 and 2
- Qubit 3 detects errors between qubits 2 and 4

Operation:
1. Initialize all qubits
2. Perform syndrome extraction (measure ancillas 1 and 3)
3. Repeat syndrome extraction 5 times
4. Final measurement of all data qubits (0, 2, 4)

Error Model:
- ex_reset: X-flip errors after qubit reset
- pd_cx: Depolarizing errors on CNOT gates (2-qubit)
- pd_idle: Depolarizing errors on idle qubits
- ex_read: X-flip errors before measurement

Output:
- Detectors track syndrome changes over time
- Observable tracks the logical qubit state (encoded in qubit 4)
- Can detect and correct single bit-flip errors on data qubits
'''

def generate_stim_circuit(num_repeat=5, ex_reset=1e-3, pd_cx=1e-3, pd_idle=1e-3, ex_read=1e-3):
    """
    Generate a Stim circuit with parameterizable error rates.
    
    Parameters:
    -----------
    ex_reset : float
        X-error probability after reset operations
    pd_cx : float
        Depolarizing error probability for 2-qubit CX gates
    pd_idle : float
        Depolarizing error probability for idle qubits
    ex_read : float
        X-error probability before measurement
    
    Returns:
    --------
    str : The Stim circuit definition
    """
    
    circuit = f"""# ====  initial state
# Reset all qubits:
R 0 1 2 3 4

# apply post-reset X-noise to each qubit, independent:
X_ERROR({ex_reset}) 0 1 2 3 4
TICK

# CNOT gates 'pointing down',  CNOT(0,1), CNOT(2,3)
CX 0 1 2 3

# apply 2q noise to each qubit touched by CX, equal chance of all 15 possible errors for a pair of qubits: IX,IY,...
DEPOLARIZE2({pd_cx}) 0 1 2 3

# apply noise to idle qubit, depolarize means p/3 of each X,Y,Z error
DEPOLARIZE1({pd_idle}) 4
TICK

# CNOT gates 'pointing up'
CX 2 1 4 3
DEPOLARIZE1({pd_idle}) 0
DEPOLARIZE2({pd_cx}) 2 1 4 3
TICK

# add error *before* measurement
X_ERROR({ex_read}) 1 3
M 1 3

# apply noise to idle qubit
DEPOLARIZE1({pd_idle}) 0 2 4

# Set coordinates (x,t) of detectors and the measurements they depend on, ex. x=qubit
# The coordinates written in the DETECTOR() command are local, in particular within repeat-loop
DETECTOR(1, 0) rec[-2]
DETECTOR(3, 0) rec[-1]


#====== repeating gates
REPEAT {num_repeat} {{
R 1 3
X_ERROR({ex_reset}) 1 3
DEPOLARIZE1({pd_idle}) 0 2 4
TICK
CX 0 1 2 3
DEPOLARIZE2({pd_cx}) 0 1 2 3
DEPOLARIZE1({pd_idle}) 4
TICK
CX 2 1 4 3
DEPOLARIZE1({pd_idle}) 0
DEPOLARIZE2({pd_cx}) 2 1 4 3
TICK
X_ERROR({ex_read}) 1 3
M 1 3
DEPOLARIZE1({pd_idle}) 0 2 4

#...... completion of repeat-loop
#  command moves that cursor by (dx, dt) immediately.
SHIFT_COORDS(0, 1)
DETECTOR(1, 0) rec[-2] rec[-4]
DETECTOR(3, 0) rec[-1] rec[-3]
}}

# = = = ==  final measurements
X_ERROR({ex_read}) 0 2 4
M 0 2 4

SHIFT_COORDS(0, 1)

#.... final detector is defined as parity of 3 measurements
DETECTOR(1, 0) rec[-2] rec[-3] rec[-5]
DETECTOR(3, 0) rec[-1] rec[-2] rec[-4]

# Choose the measurement(s) that make up the logical observables:
OBSERVABLE_INCLUDE(0) rec[-1]
"""
    
    return circuit


def write_circuit_to_file(filename, ex_reset, pd_cx, pd_idle, ex_read):
    """
    Generate circuit and write to file.
    
    Parameters:
    -----------
    filename : str
        Output filename
    ex_reset, pd_cx, pd_idle, ex_read : float
        Error rate parameters
    """
    circuit = generate_stim_circuit(ex_reset, pd_cx, pd_idle, ex_read)
    
    with open(filename, 'w') as f:
        f.write(circuit)
    
    print(f"Circuit written to {filename}")


def main():
    """Test the circuit generation with different error rates."""
    
    # Test 1: Original parameters (0.001 for all)
    print("Test 1: Generating circuit with p=0.001 for all errors")
    circuit1 = generate_stim_circuit(
        ex_reset=0.001,
        pd_cx=0.001,
        pd_idle=0.001,
        ex_read=0.001
    )
    write_circuit_to_file("circuit_test1.stim", 0.001, 0.001, 0.001, 0.001)
    
    # Test 2: Different error rates
    print("\nTest 2: Generating circuit with varied error rates")
    circuit2 = generate_stim_circuit(
        ex_reset=0.01,   # 1% reset error
        pd_cx=0.005,     # 0.5% CX error
        pd_idle=0.001,   # 0.1% idle error
        ex_read=0.002    # 0.2% readout error
    )
    write_circuit_to_file("circuit_test2.stim", 0.01, 0.005, 0.001, 0.002)
    
    # Test 3: Low error rates
    print("\nTest 3: Generating circuit with low error rates")
    circuit3 = generate_stim_circuit(
        ex_reset=0.0001,
        pd_cx=0.0001,
        pd_idle=0.0001,
        ex_read=0.0001
    )
    write_circuit_to_file("circuit_test3.stim", 0.0001, 0.0001, 0.0001, 0.0001)
    
    # Test 4: Just print to console
    print("\nTest 4: Printing circuit to console")
    print(circuit1[:500])  # Print first 500 chars
    print("...")
    
    # Optional: Verify with Stim (if stim is installed)
    try:
        import stim
        circ = stim.Circuit(circuit1)
        print(f"\nCircuit successfully parsed by Stim!")
        print(f"Number of qubits: {circ.num_qubits}")
        print(f"Number of detectors: {circ.num_detectors}")
        print(f"Number of measurements: {circ.num_measurements}")
    except ImportError:
        print("\nStim not installed - skipping validation")
    except Exception as e:
        print(f"\nError parsing circuit: {e}")


if __name__ == "__main__":
    main()
