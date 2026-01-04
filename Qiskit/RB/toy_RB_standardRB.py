#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# based on https://qiskit-community.github.io/qiskit-experiments/manuals/verification/randomized_benchmarking.html

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
# Qiskit imports
from qiskit_experiments.library import StandardRB
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeCusco

# --- 1. Construct the RB Experiment Definition ---

# Define the parameters for our RB experiment
lengths = np.arange(1, 250, 25)  # The different Clifford lengths (m)
num_samples = 10  # Number of random sequences for each length
seed = 42  # Seed for reproducibility
qubits = [3]  # We will benchmark qubit 0 of the device

print("Step 1: Defining the Randomized Benchmarking experiment...")
# This object defines the experiment but doesn't generate the circuits yet.
rb_exp = StandardRB(    qubits,    lengths=lengths,    num_samples=num_samples,    seed=seed)

# --- 2. Instantiate Backend ---
print("\nStep 2: Instantiating the noisy backend ...")
backend = FakeTorino()
backend=FakeCusco()
print(f"-> Backend '{backend.name}' is ready.")

# --- 3. Run Experiment and Await Analysis Results ---

print("\nStep 3: Running the experiment on the backend...")
# The .run() method is a high-level command that handles circuit generation,
# transpilation, execution, and triggers the analysis automatically.
exp_data = rb_exp.run(backend, seed_simulator=seed)
exp_data.block_for_results()  # Wait for the analysis to complete
print("-> Experiment and analysis finished.")
print('bb',exp_data)

fig, ax = plt.subplots( figsize=(6,8))
# Tell the plotter to use this axes
plotter = rb_exp.analysis.plotter
plotter.options.axis = ax
# Regenerate the figure on that axes
rb_exp.analysis.run(exp_data).block_for_results()
ax.set(title='Single qubit RB, %s, q=%d'%(backend.name,qubits[0]))

outF='out/rbA_%s_q%d.png'%(backend.name,qubits[0])
plt.tight_layout()
plt.savefig(outF); print('saved',outF)


# --- 4. Extract Fitted Parameters from Analysis ---
print("\nStep 4: Extracting fit results from the completed analysis...")

analysis_results = exp_data.analysis_results(dataframe=False)
k=5
for i in range(k):
    valO=analysis_results[i]
    print('obs=%s:  %s  quality=%s  chi2=%.1f  %s'%(   valO.name,valO.value,valO.quality,valO.chisq,valO.device_components))
