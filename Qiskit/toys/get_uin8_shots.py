#!/usr/bin/env python3
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.primitives.containers.bit_array import BitArray

def pack_shotsV2_to_numpy(pub_result, verb=1):
    """
    Extracts raw uint8 arrays and metadata from a SamplerV2 PubResult.
    
    Returns:
        arrays_dict (dict): {register_name: np.ndarray (uint8)}
        meta_dict (dict):   {register_name: int (num_bits)}
    """
    arrays_dict = {}
    meta_dict = {}
    
    # pub_result.data is a DataBin. We iterate over its fields (register names).
    # Since DataBin is dynamic, we inspect its standard attributes.
    for field_name in pub_result.data:
        # Get the actual data object (should be a BitArray)
        field_val = getattr(pub_result.data, field_name)
        
        # Check if it is indeed a BitArray (to ignore hidden fields/metadata)
        if isinstance(field_val, BitArray):
            arrays_dict[field_name] = field_val.array  # The raw uint8 numpy array
            meta_dict[field_name] = field_val.num_bits # The integer bit width

    if verb>0:
        for reg_name in arrays_dict:
            print(f"Register '{reg_name}', Shape: {arrays_dict[reg_name].shape}, NumBits: {meta_dict[reg_name]} , Dtype: {arrays_dict[reg_name].dtype}")

    return arrays_dict, meta_dict

def unpack_shotsV2_from_numpy(raw_uint8_array, num_bits):
    """
    Converts raw uint8 array back into a dictionary of bitstring frequencies.
    
    Args:
        raw_uint8_array (np.ndarray): The packed data (shots x bytes)
        num_bits (int): The width of the register (to slice padding)
        
    Returns:
        counts (dict): {'0011': 45, '1010': 12, ...}
    """
    # 1. Unpack bits using little-endian order (matches Qiskit convention)
    # shape becomes (shots, total_unpacked_bits)
    unpacked_bits = np.unpackbits(raw_uint8_array, axis=1, bitorder='little')
    
    # 2. Slice to the exact number of valid bits (remove padding from 8-bit blocks)
    valid_bits = unpacked_bits[:, :num_bits]
    
    # 3. Convert binary rows to bitstrings and count frequencies
    # Optimization: Convert rows to string keys for counting
    # Note: For very large arrays, unique_rows methods are faster, but this is clearest.
    counts = {}
    
    # We iterate over the shots
    for row in valid_bits:
        # Create bitstring (e.g. [1, 0, 1] -> "101")   
        # Let's stick to the raw bit order produced by unpackbits for consistency with
        # standard Qiskit 'get_bitstrings()' behavior which reverses for printing.
        # Standard Qiskit get_bitstrings() returns "q_n ... q_0".
        
        # valid_bits[0] is q_0.
        # We want string "q_n ... q_0". So we reverse the row.
        bs_str = "".join(str(b) for b in row[::-1])
        
        if bs_str in counts:
            counts[bs_str] += 1
        else:
            counts[bs_str] = 1
            
    return counts

# ==========================================
# Runnable Example
# ==========================================
if __name__ == "__main__":
    # 1. Create a Circuit with 2 Classical Registers
    #    Register 'measA': 2 bits (measure q0, q1)
    #    Register 'measB': 1 bit  (measure q2)
    qr = QuantumRegister(3)
    crA = ClassicalRegister(2, name='measA')
    crB = ClassicalRegister(1, name='measB')
    qc = QuantumCircuit(qr, crA, crB)
    
    # Create a state: |111>
    qc.x([ 1, 2])
    qc.h(2)
    
    qc.measure(qr[0], crA[0])
    qc.measure(qr[1], crA[1])
    qc.measure(qr[2], crB[0])
    
    print("--- Circuit ---")
    print(qc)

    # 2. Run with SamplerV2
    backend = AerSimulator()
    sampler = Sampler(mode=backend)
    job = sampler.run([qc], shots=100) # Run 100 shots
    pub_result = job.result()[0]
    
    print("\n--- 1. Pack Data (V2 Object -> Numpy Dicts) ---")
    arrays, metadata = pack_shotsV2_to_numpy(pub_result)
    
   
    print("\n--- 2. Unpack Data (Numpy -> Counts Dict) ---")
    for reg_name in arrays:
        raw_data = arrays[reg_name]
        n_bits   = metadata[reg_name]
        
        counts = unpack_shotsV2_from_numpy(raw_data, n_bits)
        print(f"Counts for '{reg_name}': {counts}")
