# based on https://arxiv.org/abs/2408.15653v1

# --- Imports ---
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Qiskit imports for gate-based simulation
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator

# --- Parameters ---
START_NODE = 7
MAX_STEPS = 15 # Number of steps to simulate

# The specified graph with 16 nodes and 32 edges
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

# --- 1. Graph Definition ---
G = nx.Graph()
G.add_weighted_edges_from(graph_edges)
for i in range(16):
    if i not in G: G.add_node(i)
for node in G.nodes():
    G.nodes[node]['weight'] = 1.0

N = G.number_of_nodes()
sorted_nodes = sorted(G.nodes())

# --- 2. Qubit and State Space Definition ---
num_node_qubits = int(np.ceil(np.log2(N)))
num_edge_qubits = int(np.ceil(np.log2(2 * G.number_of_edges())))
total_qubits = num_node_qubits + num_edge_qubits
print(f'Graph: {N} nodes, {G.number_of_edges()} edges')
print(f'Qubits: {num_node_qubits} (node) + {num_edge_qubits} (edge) = {total_qubits} total')

edge_to_int = {}
int_to_edge = {}
edge_counter = 0
for u in sorted(G.nodes()):
    for v in sorted(G.neighbors(u)):
        edge_to_int[(u, v)] = edge_counter
        int_to_edge[edge_counter] = (u, v)
        edge_counter += 1

# --- 3. Operator Implementation (as NumPy matrices) ---
def create_coin_matrix():
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

def create_shift_matrix():
    dim = 2**total_qubits
    shift_matrix = np.zeros((dim, dim), dtype=complex)
    for state_idx in range(dim):
        node_val = state_idx & ((1 << num_node_qubits) - 1)
        edge_val = state_idx >> num_node_qubits
        if edge_val in int_to_edge:
            u, v = int_to_edge[edge_val]
            if u == node_val:
                rev_edge_val = edge_to_int[(v, u)]
                new_state_idx = (rev_edge_val << num_node_qubits) | v
                shift_matrix[new_state_idx, state_idx] = 1
            else:
                shift_matrix[state_idx, state_idx] = 1
        else:
            shift_matrix[state_idx, state_idx] = 1
    return shift_matrix

# --- 4. MSD Calculation Functions ---
def get_quantum_msd_qiskit(coin_op, shift_op, initial_state, distances_sq):
    """Calculates MSD by building and simulating a Qiskit circuit for each step."""
    msd_values = []
    backend = AerSimulator(method='statevector')
    
    # Define the qubit registers based on the paper's structure
    qr_node = QuantumRegister(num_node_qubits, name='node')
    qr_edge = QuantumRegister(num_edge_qubits, name='edge')
    all_qubits = list(qr_node) + list(qr_edge)

    for t in range(1, MAX_STEPS + 1):
        # Build the circuit for 't' steps
        qc = QuantumCircuit(qr_node, qr_edge)
        #1qc.initialize(initial_state, all_qubits)
        qc.barrier()
        
        # Append the operators for each step
        for _ in range(t):
            qc.append(coin_op, all_qubits)
            qc.append(shift_op, all_qubits)
        
        # Save the statevector
        qc.save_statevector()
        if t==1 :
            print(qc)
            qcT= transpile(qc, backend=backend, optimization_level=3, basis_gates=['u','cz'])
            print('Transpiled, ops:',qcT.count_ops())
            
            # Simulate to get the final statevector
        job = backend.run(qc)
        result = job.result()
        final_statevector = result.get_statevector()

        # Calculate node probabilities from the final statevector
        node_probs = np.zeros(N)
        for i in range(N):
            amplitudes_i = [final_statevector.data[idx] for idx in range(2**total_qubits) if (idx & ((1 << num_node_qubits) - 1)) == i]
            node_probs[i] = np.sum(np.abs(amplitudes_i)**2)
        
        msd = np.sum(node_probs * distances_sq)
        msd_values.append(msd)
        print(f"  Qiskit Sim Step {t}/{MAX_STEPS}: MSD = {msd:.4f}")
        
    return msd_values

def get_classical_msd(M_step, initial_state, distances_sq):
    msd_values = []
    current_state = initial_state
    for step in range(MAX_STEPS):
        current_state = M_step @ current_state
        msd = np.sum(current_state * distances_sq)
        msd_values.append(msd)
    return msd_values

# --- 5. Main Simulation ---
if __name__ == '__main__':
    print("Preparing graph and operators...")
    distances = np.array([nx.shortest_path_length(G, source=START_NODE, target=i) for i in sorted_nodes])
    distances_sq = distances**2

    # --- Quantum Walk Setup ---
    # Create matrices and wrap them as Qiskit Operator objects
    coin_op = Operator(create_coin_matrix())
    shift_op = Operator(create_shift_matrix())

    # Create initial state vector
    psi_0 = np.zeros(2**total_qubits, dtype=complex)
    start_neighbors = sorted(G.neighbors(START_NODE))
    k_start = len(start_neighbors)
    if k_start > 0:
        amplitude = 1.0 / np.sqrt(k_start)
        for neighbor in start_neighbors:
            edge_val = edge_to_int[(START_NODE, neighbor)]
            state_idx = (edge_val << num_node_qubits) | START_NODE
            psi_0[state_idx] = amplitude
    
    # --- Classical Walk Setup ---
    A = nx.to_numpy_array(G, nodelist=sorted_nodes)
    degrees = A.sum(axis=1)
    M_step = (A / degrees[:, np.newaxis]).T
    p_0 = np.zeros(N)
    p_0[START_NODE] = 1.0

    # --- Run Calculations ---
    print(f"Calculating Quantum MSD for {MAX_STEPS} steps using Qiskit...")
    q_msd = get_quantum_msd_qiskit(coin_op, shift_op, psi_0, distances_sq)
    
    print(f"\nCalculating Classical MSD for {MAX_STEPS} steps...")
    c_msd = get_classical_msd(M_step, p_0, distances_sq)
    
    # --- Plotting ---
    print("\nGenerating plot...")
    steps = np.arange(1, MAX_STEPS + 1)
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(steps, q_msd, 'o-', color='darkslateblue', label='Quantum Walk MSD (Qiskit Sim)')
    ax.plot(steps, c_msd, 's-', color='firebrick', label='Classical Walk MSD')

    popt_q, _ = curve_fit(lambda t, a: a * t**2, steps, q_msd)
    ax.plot(steps, popt_q[0] * steps**2, '--', color='cornflowerblue', label=r'Quantum Fit ($\propto t^2$)')

    popt_c, _ = curve_fit(lambda t, b: b * t, steps, c_msd)
    ax.plot(steps, popt_c[0] * steps, '--', color='salmon', label=r'Classical Fit ($\propto t$)')

    ax.set_xlabel("Number of Steps (t)", fontsize=12)
    ax.set_ylabel("Mean Squared Displacement (MSD)", fontsize=12)
    ax.set_title(f"Quantum vs. Classical Walk Transport Efficiency (Start Node {START_NODE})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    outF = 'outQw_MSD_Qiskit_Sim.png'
    fig.savefig(outF)
    print(f'saved: {outF}')
    plt.show()
