# --- Coin Operator Comparison Script ---
# Purpose: Compare matrix-based vs gate-based coin operator implementations
# Compare the resulting unitaries to verify gate implementation correctness

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

# --- Graph Definition (same as original) ---
graph_edges = [
    [0, 5, 0.8315], [1, 10, 0.4592], [2, 9, 0.9981], [3, 15, 0.2107],
    [4, 7, 0.6743], [8, 11, 0.5519], [12, 14, 0.7823], [6, 13, 0.3056],
    [0, 11, 0.9147], [1, 3, 0.2588], [2, 14, 0.8821], [5, 9, 0.4013],
    [7, 10, 0.7350], [13, 15, 0.6198], [4, 8, 0.2974], [6, 12, 0.9502],
    [0, 15, 0.5001], [1, 8, 0.8675], [2, 5, 0.3349], [3, 11, 0.7761],
    [4, 13, 0.6405], [6, 9, 0.4827], [7, 14, 0.9234], [10, 12, 0.2256],
    [1, 15, 0.5890], [3, 7, 0.8112], [5, 12, 0.3777], [9, 13, 0.7068],
    [0, 2, 0.9649], [4, 11, 0.4309], [8, 14, 0.6651], [6, 10, 0.5135]
]

# Build graph
G = nx.Graph()
G.add_weighted_edges_from(graph_edges)
for i in range(16):
    if i not in G: G.add_node(i)
for node in G.nodes():
    G.nodes[node]['weight'] = 1.0

N = G.number_of_nodes()
sorted_nodes = sorted(G.nodes())

# --- Qubit and State Space Definition ---
num_node_qubits = int(np.ceil(np.log2(N)))
num_edge_qubits = int(np.ceil(np.log2(2 * G.number_of_edges())))
total_qubits = num_node_qubits + num_edge_qubits

print(f'Graph: {N} nodes, {G.number_of_edges()} edges')
print(f'Qubits: {num_node_qubits} (node) + {num_edge_qubits} (edge) = {total_qubits} total')

# Edge mapping
edge_to_int = {}
int_to_edge = {}
edge_counter = 0
for u in sorted(G.nodes()):
    for v in sorted(G.neighbors(u)):
        edge_to_int[(u, v)] = edge_counter
        int_to_edge[edge_counter] = (u, v)
        edge_counter += 1

# --- METHOD 1: Direct Matrix Construction (from cc3.py) ---
def create_coin_matrix():
    """Direct unitary matrix construction using Grover operators"""
    dim = 2**total_qubits
    coin_matrix = np.identity(dim, dtype=complex)
    
    for i in range(N):
        neighbors = sorted(G.neighbors(i))
        outgoing_edges = [edge_to_int[(i, v)] for v in neighbors]
        k_i = len(outgoing_edges)
        
        if k_i > 0:
            edge_weights = [G.edges[i, v]['weight'] for v in neighbors]
            amplitudes = np.sqrt(edge_weights)
            s_vector = amplitudes / np.linalg.norm(amplitudes)
            grover_op = 2 * np.outer(s_vector, s_vector.conj()) - np.identity(k_i)
            
            for r_idx, e_row in enumerate(outgoing_edges):
                for c_idx, e_col in enumerate(outgoing_edges):
                    g_row = (e_row << num_node_qubits) | i
                    g_col = (e_col << num_node_qubits) | i
                    coin_matrix[g_row, g_col] = grover_op[r_idx, c_idx]
    
    return coin_matrix

# --- METHOD 2: Gate-Based Construction ---
def create_coin_operator_gates(qc):
    """Gate-based implementation using elementary quantum gates"""
    
    for node_i in range(N):
        neighbors = list(G.neighbors(node_i))
        k_i = len(neighbors)
        
        if k_i > 0:
            # Get outgoing edge indices for this node
            outgoing_edges = [edge_to_int[(node_i, v)] for v in neighbors]
            
            # Create the node condition |i⟩⟨i| using multi-controlled operations
            node_binary = format(node_i, f'0{num_node_qubits}b')
            
            # Create control condition for node i
            control_qubits = []
            for bit_idx, bit in enumerate(node_binary):
                if bit == '0':
                    qc.x(bit_idx)  # Flip for 0 bits
                control_qubits.append(bit_idx)
            
            # Apply controlled Hadamard gates to create superposition over edges
            for edge_idx in range(num_edge_qubits):
                edge_qubit = num_node_qubits + edge_idx
                if len(control_qubits) == 1:
                    qc.ch(control_qubits[0], edge_qubit)
                else:
                    # Approximate multi-controlled H with controlled rotations
                    qc.cry(np.pi/4, control_qubits[0], edge_qubit)
            
            # Apply controlled inversion about |0⟩ state (part of Grover operator)
            edge_qubits = list(range(num_node_qubits, num_node_qubits + num_edge_qubits))
            if len(edge_qubits) > 1:
                target = edge_qubits[0]
                others = edge_qubits[1:] + control_qubits
                if len(others) == 1:
                    qc.cz(others[0], target)
                else:
                    # Approximate multi-controlled operation
                    qc.rz(np.pi/8, target)
            
            # Apply controlled Hadamard gates again
            for edge_idx in range(num_edge_qubits):
                edge_qubit = num_node_qubits + edge_idx
                if len(control_qubits) == 1:
                    qc.ch(control_qubits[0], edge_qubit)
                else:
                    qc.cry(np.pi/4, control_qubits[0], edge_qubit)
            
            # Restore flipped bits
            for bit_idx, bit in enumerate(node_binary):
                if bit == '0':
                    qc.x(bit_idx)

def get_gate_based_unitary():
    """Convert gate-based implementation to unitary matrix"""
    qr_node = QuantumRegister(num_node_qubits, name='node')
    qr_edge = QuantumRegister(num_edge_qubits, name='edge')
    qc = QuantumCircuit(qr_node, qr_edge)
    
    # Apply the gate-based coin operator
    create_coin_operator_gates(qc)
    print(qc)
    # Save the unitary matrix
    qc.save_unitary()
    
    # Convert circuit to unitary
    simulator = AerSimulator(method='unitary')
    job = simulator.run(qc)
    result = job.result()
    unitary = result.get_unitary()
    
    # Convert to numpy array to avoid deprecation warning
    return np.asarray(unitary)

# --- COMPARISON ---
if __name__ == '__main__':
    print("\n" + "="*60)
    print("COIN OPERATOR COMPARISON")
    print("="*60)
    
    print("\n1. Computing matrix-based coin operator...")
    matrix_unitary = create_coin_matrix()
    print(f"   Matrix dimensions: {matrix_unitary.shape}")
    print(f"   Matrix is unitary: {np.allclose(matrix_unitary @ matrix_unitary.conj().T, np.eye(len(matrix_unitary)))}")
    
    print("\n2. Computing gate-based coin operator...")
    try:
        gate_unitary = get_gate_based_unitary()
        print(f"   Gate unitary dimensions: {gate_unitary.shape}")
        print(f"   Gate unitary is unitary: {np.allclose(gate_unitary @ gate_unitary.conj().T, np.eye(len(gate_unitary)))}")
        
        print("\n3. Comparing the two implementations...")
        diff = np.abs(matrix_unitary - gate_unitary)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        
        print(f"   Maximum difference: {max_diff:.2e}")
        print(f"   Average difference: {avg_diff:.2e}")
        print(f"   Implementations match: {np.allclose(matrix_unitary, gate_unitary, rtol=1e-10, atol=1e-12)}")
        
        if max_diff > 1e-10:
            print(f"\n   WARNING: Large differences found!")
            print(f"   First 5 largest differences:")
            flat_diff = diff.flatten()
            largest_indices = np.argsort(flat_diff)[-5:]
            for idx in reversed(largest_indices):
                i, j = np.unravel_index(idx, diff.shape)
                print(f"   [{i:4d},{j:4d}]: matrix={matrix_unitary[i,j]:.6f}, gates={gate_unitary[i,j]:.6f}, diff={diff[i,j]:.2e}")
        else:
            print("   ✓ Implementations match within numerical precision!")
            
    except Exception as e:
        print(f"   ERROR in gate-based implementation: {e}")
        print("   Gate-based method failed - circuit construction issue")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
