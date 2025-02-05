#!/usr/bin/env python3
# problem: display 'experiment data figure' w/ pyplot, solved
# ref code:  https://qiskit.org/documentation/experiments/tutorials/randomized_benchmarking.html

import matplotlib.pyplot as plt

import numpy as np
from qiskit_experiments.library import StandardRB, InterleavedRB
from qiskit_experiments.framework import ParallelExperiment, BatchExperiment
import qiskit.circuit.library as circuits

# For simulation
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeLima

backend = AerSimulator.from_backend(FakeLima())

lengths = np.arange(1, 800, 200)
num_samples = 10
seed = 1010
qubits = [0]

# Run an RB experiment on qubit 0
exp1 = StandardRB(qubits, lengths, num_samples=num_samples, seed=seed)
expdata1 = exp1.run(backend).block_for_results()
results1 = expdata1.analysis_results()
for result in results1:
    print(result)
    #aa=result.to_dict()
    #print('result as dict:',aa)
    
#.....  display only
def show_figure(figPayload):  # from Aziz, IBM Quantum Support
    print('SF:figPayload=',type(figPayload))
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = figPayload
    figPayload.set_canvas(new_manager.canvas)

rbFitFig = expdata1.figure(0).figure
ax= rbFitFig.axes[0]
ax.set_xlim(0.9,)
ax.set_xscale('log')
#print('aa',ax_list)
show_figure(rbFitFig)

plt.show()
