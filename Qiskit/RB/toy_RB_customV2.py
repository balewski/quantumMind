#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
This script performs a single-qubit Randomized Benchmarking (RB) experiment from scratch
using Qiskit. It builds circuits using native gates based on Clifford name strings.

The script executes the following steps:
1.  **Generate Clifford Group:** Programmatically creates the set of 24 single-qubit
    Clifford gates and their string representations (e.g., "I", "HS", "SH").
2.  **Construct RB Sequences:** For a range of sequence lengths 'm', it generates multiple
    random sequences of Clifford gates. Circuits are built by parsing the name strings
    and applying native Qiskit gates (I, H, S). The inverse is computed using Clifford
    algebra and converted back to a name string.
3.  **Define Backend:** Instantiates a noisy simulated backend (`FakeTorino`) from
    `qiskit_ibm_runtime.fake_provider`.
4.  **Transpile and Execute:** The generated circuits are transpiled for the specific
    backend and target qubit. The job is then run on the simulator.
5.  **Analyze and Fit Results:** All raw data is processed. The average survival
    probabilities are fitted to the exponential decay model P(m) = A * p^m + B.
6.  **Print Summary:** A consolidated summary of all calculated results is printed.
7.  **Plot Results:** The final results are visualized with violin plots, fitted curve,
    and detailed statistics.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import os

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeCusco
from qiskit.quantum_info import Clifford
from qiskit.circuit.library import HGate, SGate, SdgGate, XGate, YGate, ZGate, IGate

# --- Helper Function to Generate the Single-Qubit Clifford Group ---
def generate_single_qubit_clifford_group():
    """
    Generates the 24 single-qubit Clifford gates using the group generators H and S.
    This method is robust and guarantees the generation of the complete group.
    It also creates names for each unique Clifford gate.
    Returns a tuple of (unique_cliffords, names).
    """
    cliff_h = Clifford.from_circuit(HGate())
    cliff_s = Clifford.from_circuit(SGate())
    cliff_i = Clifford(np.eye(2)) 

    unique_cliffords = [cliff_i]
    names = ["I"]  # Name for Identity gate
    queue = [(cliff_i, "I")]  # Queue of tuples (Clifford, name)
    
    while queue:
        current_cliff, current_name = queue.pop(0)
        
        # Define generators
        generators = [(cliff_h, 'H'), (cliff_s, 'S')]
        
        for gen, gate_name in generators:
            new_cliff = current_cliff.compose(gen)
            new_name = current_name + gate_name
            
            if new_cliff not in unique_cliffords:
                unique_cliffords.append(new_cliff)
                names.append(new_name)  # Append the new name as well
                queue.append((new_cliff, new_name))

    if len(unique_cliffords) != 24:
        print(f"Warning: Generated {len(unique_cliffords)} Cliffords, expected 24. There is an issue in the generation logic.")

    return unique_cliffords, names


def apply_clifford_name_to_circuit(qc, qubit_idx, clifford_name):
    """
    Applies a Clifford gate to a circuit based on its name string.
    Parses the name character by character and applies corresponding gates.
    
    Args:
        qc: QuantumCircuit to modify
        qubit_idx: index of qubit to apply gates to
        clifford_name: string like "I", "H", "HS", "SHS", etc.
    """
    gate_map = {
        'I': IGate(),
        'H': HGate(),
        'S': SGate()
    }
    
    for char in clifford_name:
        if char in gate_map:
            qc.append(gate_map[char], [qubit_idx])
        else:
            raise ValueError(f"Unknown gate character: {char}")


def find_clifford_name_from_matrix(clifford_obj, all_cliffords, all_names):
    """
    Find the name of a Clifford by comparing it to all known Cliffords.
    
    Args:
        clifford_obj: Clifford object to identify
        all_cliffords: list of all 24 Clifford objects
        all_names: list of corresponding names
    
    Returns:
        Name string of the matching Clifford
    """
    for i, cliff in enumerate(all_cliffords):
        if clifford_obj == cliff:
            return all_names[i]
    
    raise ValueError("Clifford not found in the group!")


# --- Step 1: Construct List of RB Gate Sequences Manually ---
print("Step 1: Generating Randomized Benchmarking sequences from scratch...")
qubit_to_test = 3
lengths = np.arange(1, 250, 25)
num_samples = 10
seed = 42
random.seed(seed)
np.random.seed(seed)

single_qubit_cliffords, cliffords_names = generate_single_qubit_clifford_group()

print(f"-> Generated {len(single_qubit_cliffords)} Cliffords")
print("-> Clifford names:", cliffords_names[:10], "...")

rb_circuits = []
print(f"-> Generating circuits for {len(lengths)} lengths and {num_samples} samples each...")
for m in lengths:
    print('\nm=',m)
    for i in range(num_samples):
        # Track the combined Clifford using Clifford algebra
        tracking_clifford = Clifford(np.eye(2))
        qc = QuantumCircuit(1, 1)
        
        for _ in range(m):
            rand_cliff_idx = random.randint(0, len(single_qubit_cliffords) - 1)
            random_clifford = single_qubit_cliffords[rand_cliff_idx]
            random_clifford_name = cliffords_names[rand_cliff_idx]
            
            if i==0: 
                print(f'{rand_cliff_idx}:{random_clifford_name}, ', end='')
            
            # Apply the Clifford by parsing its name string
            apply_clifford_name_to_circuit(qc, 0, random_clifford_name)
            qc.barrier()
            
            # Track composition for inverse calculation
            tracking_clifford = tracking_clifford.compose(random_clifford)
        
        # Compute inverse and find its name
        inverse_clifford = tracking_clifford.adjoint()
        inverse_name = find_clifford_name_from_matrix(inverse_clifford, single_qubit_cliffords, cliffords_names)
        
        if i==0:
            print(f'| inv:{inverse_name}')
        
        # Apply inverse using its name
        apply_clifford_name_to_circuit(qc, 0, inverse_name)
        
        qc.measure(0, 0)
        qc.metadata = {'xval': m}
        rb_circuits.append(qc)

print(f"-> Successfully generated {len(rb_circuits)} circuits.")

# --- Step 2: Instantiate Backend ---
print("\nStep 2: Instantiating the noisy backend ...")
backend = FakeTorino()
#backend=FakeCusco()
print(f"-> Backend '{backend.name}' is ready.")

# --- Step 3: Transpile, Run, and Collect Raw Data ---
print("\nStep 3: Transpiling and running circuits on the backend...")
print("-> Transpiling circuits for the backend's basis gates...")
transpiled_circuits = transpile(rb_circuits, backend=backend, optimization_level=1, initial_layout=[qubit_to_test])
print("-> Transpilation complete.")

job = backend.run(transpiled_circuits, shots=1024, seed_simulator=seed)
result = job.result()
print("-> Job finished.")

results_by_length = {}
for i, circuit in enumerate(rb_circuits):
    m = circuit.metadata['xval']
    counts = result.get_counts(i)
    survival_prob = counts.get('0', 0) / sum(counts.values())
    if m in results_by_length:
        results_by_length[m].append(survival_prob)
    else:
        results_by_length[m] = [survival_prob]
print("-> Successfully collected raw results.")

# --- Step 4: Perform Data Analysis and Fitting ---
print("\nStep 4: Analyzing data and fitting decay curve...")
m_values = np.array(sorted(list(results_by_length.keys())))
avg_probs = np.array([np.mean(results_by_length[m]) for m in m_values])
std_errs = np.array([np.std(results_by_length[m]) / np.sqrt(len(results_by_length[m])) for m in m_values])

def rb_decay(m, A, p, B):
    return A * (p ** m) + B

fit_successful = False
try:
    popt, pcov = curve_fit(rb_decay, m_values, avg_probs, p0=[0.5, 0.99, 0.5], sigma=std_errs, absolute_sigma=True, maxfev=5000)
    
    A_fit, p_fit, B_fit = popt
    perr = np.sqrt(np.diag(pcov))
    A_err, p_err, B_err = perr

    n_qubits = 1   # Assuming a single qubit
    d = 2 ** n_qubits
 
    epc = (d - 1) / d * (1 - p_fit)
    epc_err = (d - 1) / d * p_err
    
    y_fit_at_points = rb_decay(m_values, A_fit, p_fit, B_fit)
    chi2 = np.sum(((avg_probs - y_fit_at_points) / std_errs)**2)
    dof = len(m_values) - len(popt)
    reduced_chi2 = chi2 / dof
    
    fit_successful = True
    print("-> Analysis complete.")

except RuntimeError as e:
    print(f"-> Fit failed: {e}. Cannot perform full analysis.")

# --- Step 5: Print Analysis Summary ---
print("\nStep 5: Consolidated Analysis Results")
print("-" * 40)
if fit_successful:
    print(f"  Fitted A          = {A_fit:.4f} +/- {A_err:.4f}")
    print(f"  Fitted B          = {B_fit:.4f} +/- {B_err:.4f}")
    print(f"  Fitted p (Fidelity) = {p_fit:.4f} +/- {p_err:.4f}")
    print(f"  Error Per Clifford (EPC) = {epc:.3e} +/- {epc_err:.3e}")
    print(f"  Goodness of Fit (χ²/ν) = {reduced_chi2:.2f}")
else:
    print("  Fit was not successful. No results to display.")
print("-" * 40)

# --- Step 6: Plot Data and Fit ---
print("\nStep 6: Plotting the data and saving the figure...")
fig, ax = plt.subplots(figsize=(6, 6))

# Create violin plots
data_for_violins = [results_by_length[m] for m in m_values]
parts = ax.violinplot(data_for_violins, positions=m_values, widths=15, showmeans=False, showmedians=False, showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('lightblue'); pc.set_edgecolor('grey'); pc.set_alpha(0.6)

# Plot average data points
ax.scatter(m_values, avg_probs, marker='o', color='royalblue', zorder=3, label='Mean Survival Probability')

# Plot the fit if successful
if fit_successful:
    m_fit_space = np.linspace(min(m_values), max(m_values), 200)
    y_fit = rb_decay(m_fit_space, *popt)
    
    # Format EPC for scientific notation in the legend
    exponent = np.floor(np.log10(abs(epc)))
    mantissa = epc / (10**exponent)
    mantissa_err = epc_err / (10**exponent)
    
    fit_label = (
        f'Fit: $A \\cdot p^m + B$\n'
        f'$A$ = {A_fit:.3f} $\\pm$ {A_err:.3f}\n'
        f'$B$ = {B_fit:.3f} $\\pm$ {B_err:.3f}\n'
        f'$p$ = {p_fit:.4f} $\\pm$ {p_err:.4f}\n'
        f'EPC = ({mantissa:.2f} $\\pm$ {mantissa_err:.2f})$\\times 10^{{{exponent:.0f}}}$\n'
        f'$\\chi^2_\\nu$ = {reduced_chi2:.2f}'
    )
    ax.plot(m_fit_space, y_fit, color='red', linestyle='--', linewidth=2, label=fit_label)

ax.set_xlabel("Clifford Length (m)", fontsize=12)
ax.set_ylabel("Ground State Survival Probability P(0)", fontsize=12)
ax.set_title(f"Single-Qubit RB with Native Gates on {backend.name} (Qubit {qubit_to_test})", fontsize=14)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(fontsize=10)

# Save the plot
output_dir = 'out'
os.makedirs(output_dir, exist_ok=True)
outF = os.path.join(output_dir, f'rbB_{backend.name}_q{qubit_to_test}V2.png')
plt.savefig(outF, dpi=300)
print(f"-> Plot saved as '{outF}'")
plt.show()
