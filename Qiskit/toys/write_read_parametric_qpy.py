#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Comprehensive workflow for creating, serializing, and deserializing parametric quantum circuits with metadata

================================================================================
Qiskit Circuit and Metadata Serialization Workflow
================================================================================

Purpose:
--------
This script demonstrates a comprehensive and robust workflow for creating, managing,
and persisting complex quantum circuits using Qiskit.  Meta data contain numpy arrays,
circuit are parametric, initial state is also parametric.

Workflow Overview:
------------------
1.  **Circuit Creation (`make_circuits_with_metadata`):**
    -   Three distinct types of quantum circuits are programmatically generated:
        1. A static Bell State circuit.
        2. A "template" circuit for parametric state initialization, which includes
           a QFT gate before being saved.
        3. A circuit with parametric rotation gates using `qiskit.circuit.Parameter`.
    -   Each circuit is assigned a rich metadata dictionary to its `.metadata`
      attribute. This metadata includes identifiers, type information, and complex
      data structures like NumPy arrays for parameters and data.

2.  **Serialization (`write_circuits_to_qpy`):**
    -   The entire list of circuit objects is serialized into a single binary
      `.qpy` file.
    -   A critical compatibility step is performed: all NumPy arrays within the
      metadata are converted to standard Python lists (`.tolist()`) before saving.
      This avoids a `TypeError` as older QPY versions use a JSON serializer
      for metadata which cannot handle the `ndarray` type.

3.  **Deserialization (`read_circuits_from_qpy`):**
    -   The script reads the `.qpy` file and deserializes it back into a list of
      Qiskit `QuantumCircuit` objects, with all metadata preserved.

4.  **Processing and Unpacking (`main` loop and helper functions):**
    -   The script iterates through the loaded circuits. For each one, it first
      reconstructs the NumPy arrays from the Python lists in the metadata.
    -   It then inspects a 'type' key in the metadata to determine how to
      process the circuit.
    -   Based on the type, it calls a dedicated processing function:
        -   `process_parametric_gates_circuit`: Assigns the numerical values
          from the metadata to the symbolic `Parameter` objects using
          `.assign_parameters()`.
        -   `process_parametric_initialize_circuit`: Dynamically builds a state
          initialization circuit from metadata and prepends it to the loaded
          QFT template using `.compose()`.
    -   The final, executable circuits are printed to demonstrate the success of
      the entire workflow.


"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import QFTGate
from qiskit import qpy
from typing import List, Dict, Any
import datetime

def make_circuits_with_metadata() -> List[QuantumCircuit]:
    """
    Creates a list of circuits with custom parametric features and metadata.
    The QFT is now included in Circuit 2 before saving.
    """
    print("--- 1. Creating circuits with parametric features and full metadata... ---")
    
    # --- Circuit 1 (2 qubits): Standard Bell State ---
    nq1 = 2
    qc1 = QuantumCircuit(nq1, name="BellState_2q")
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.measure_all()
    qc1.metadata = {
        "experiment_id": "EXP-001", "author": "Gemini1", "type": "static",
        "x": np.random.rand(2**nq1).tolist(), "y": np.random.rand(2**nq1).tolist()
    }

    # --- Circuit 2 (3 qubits): Template with QFT for Parametric Initialization ---
    nq2 = 3
    # This circuit now contains the QFT. It's a template awaiting an initial state.
    qc2 = QuantumCircuit(nq2, name="ParametricInit_QFT_3q")
    
    qft_gate = QFTGate(nq2)
    qft_gate.name = 'QFT'
    qc2.append(qft_gate, range(nq2))  # QFT is added here, before saving

    amplitudes = np.random.rand(2**nq2)
    amplitudes /= np.linalg.norm(amplitudes)
    qc2.metadata = {
        "experiment_id": "EXP-002", "author": "Gemini", "type": "parametric_initialize",
        "initial_state_amplitudes": amplitudes.tolist(),
        "x": np.random.rand(2**nq2).tolist(), "y": np.random.rand(2**nq2).tolist()
    }

    # --- Circuit 3 (4 qubits): Parametric Ry Gates ---
    nq3 = 4
    qc3 = QuantumCircuit(nq3, name="VectorizedParam_4q")
    
    # Define a standalone parameter for qubit 0
    alpha = Parameter('α')
    # Define a vector of 3 parameters for qubits 1, 2, 3
    theta = ParameterVector('θ', length=3)
    
    # Apply the gates
    qc3.ry(alpha, 0) # Apply Ry(alpha) on q0
    # Apply the vectorized Ry(theta) gates on q1, q2, q3
    for i in range(3):
        qc3.ry(theta[i], i + 1)
    qc3.cx(0, 1)     # Some entanglement for structure
    qc3.measure_all()
    
    # Generate and store values for both alpha and the theta vector
    alpha_value = np.random.uniform(0, np.pi)
    theta_values = np.random.uniform(0, 2 * np.pi, 3)
    qc3.metadata = {
        "experiment_id": "EXP-003", "author": "Gemini", "type": "parametric_gates",
        "alpha_value": alpha_value,            # Store the single float value
        "theta_values": theta_values.tolist(), # Store the list of vector values
        "x": np.random.rand(2**nq3).tolist(), "y": np.random.rand(2**nq3).tolist()
    }
    
    return [qc1, qc2, qc3]

def write_circuits_to_qpy(circuit_list: List[QuantumCircuit], filename: str):
    """Serializes a list of QuantumCircuit objects to a .qpy file."""
    print(f"\n--- 2. Writing {len(circuit_list)} circuits to '{filename}'... ---")
    with open(filename, 'wb') as file:
        qpy.dump(circuit_list, file)
    print("Save complete.")

def read_circuits_from_qpy(filename: str) -> List[QuantumCircuit]:
    """Deserializes circuits and their metadata from a .qpy file."""
    print(f"\n--- 3. Reading circuits and their metadata from '{filename}'... ---")
    with open(filename, 'rb') as file:
        loaded_circuits = qpy.load(file)
    print("Load complete.")
    return loaded_circuits

# --- Processing Functions ---

def process_parametric_gates_circuit(qc: QuantumCircuit):
    """Processes a loaded circuit with unbound Parameter and ParameterVector objects."""
    print("\n>>> Parametric Version (with unbound parameters):")
    print(qc)
    
    # Build the parameter map from the metadata
    parameter_map = {}
    
    # Find all parameters in the circuit
    params = qc.parameters
    
    # Get the values from the metadata
    alpha_val = qc.metadata['alpha_value']
    theta_vals = qc.metadata['theta_values']
    
    # Map the values to the correct Parameter objects
    for p in params:
        if p.name == 'α':
            parameter_map[p] = alpha_val
        elif p.name.startswith('θ'):
            # Extracts the index, e.g., 'θ[1]' -> 1
            index = int(p.name.split('[')[1].replace(']', ''))
            parameter_map[p] = theta_vals[index]
            
    bound_circuit = qc.assign_parameters(parameter_map)
    print("\n>>> Executable Version (parameters assigned):")
    print(bound_circuit)

def process_parametric_initialize_circuit(qc: QuantumCircuit):
    """Processes a loaded template by prepending an initialization circuit."""
    print("\n>>> Parametric Template (includes QFT, awaiting initial state):")
    print(qc)
    
    # Step 1: Create a temporary circuit with just the initialization
    num_qubits = qc.num_qubits
    amplitudes = qc.metadata['initial_state_amplitudes']
    init_circuit = QuantumCircuit(num_qubits, name="init")
    init_circuit.initialize(amplitudes, range(num_qubits))
    
    # Step 2: Compose the circuits: init_circuit followed by the loaded qc (QFT)
    final_circuit = init_circuit.compose(qc)
    final_circuit.measure_all()
    final_circuit.name = "InitializedState_QFT_3q"
    
    print("\n>>> Executable Version (dynamically composed Initialize + QFT):")
    print(final_circuit)

# --- Main Execution ---

def main():
    """Main function to demonstrate the entire workflow."""
    qpy_filename = "many_circ_with_meta.qpy"
    qcL = make_circuits_with_metadata()
    write_circuits_to_qpy(qcL, qpy_filename)
    qcL2 = read_circuits_from_qpy(qpy_filename)

    print("\n--- 4. Unpacking and Processing Loaded Data ---")
    if qcL2:
        for qc in qcL2:
            print(f"\n{'-'*50}\nProcessing Loaded Circuit: {qc.name}")
            meta = qc.metadata
            # Reconstruct numpy arrays from lists
            if meta:
                meta['x'] = np.array(meta.get('x', []))
                meta['y'] = np.array(meta.get('y', []))
            
            print("Associated Metadata:")
            for key, value in meta.items():
                print(f"  {key}: numpy array of shape {value.shape}" if isinstance(value, np.ndarray) else f"  {key}: {value}")

            # Call the appropriate processing function based on the metadata type
            if meta.get('type') == 'parametric_gates':
                process_parametric_gates_circuit(qc)
            elif meta.get('type') == 'parametric_initialize':
                process_parametric_initialize_circuit(qc)
            else:
                print("\n>>> Static Circuit:")
                print(qc)
    else:
        print("No data was loaded.")

    print(f"\n--- All circuots are saved in .qpy  --> {qpy_filename}\n")
    
if __name__ == "__main__":
    main()
