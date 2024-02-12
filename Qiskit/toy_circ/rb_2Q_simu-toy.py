#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Randomized Benchmarking
based on 
https://community.qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html
 AND
 https://qiskit.org/documentation/tutorials/noise/4_randomized_benchmarking.html

'''
# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint

#=================================
#=================================
#  M A I N
#=================================
#=================================

import numpy as np
import matplotlib.pyplot as plt

#Import the RB Functions
import qiskit.ignis.verification.randomized_benchmarking as rb

#Import Qiskit classes 
from qiskit import transpile, Aer, assemble
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, thermal_relaxation_error
import qiskit
qiskitVer=qiskit.__qiskit_version__
print('qiskit ver=',qiskitVer)
assert qiskitVer['qiskit'] >= '0.23.2'

'''
****** Step 1: Generate RB sequences *********
The RB sequences consist of random Clifford elements chosen uniformly from 
the Clifford group on  n-qubits, including a computed reversal element, 
that should return the qubits to the initial state.

More precisely, for each length  m, we choose  𝐾_m  RB-sequences. Each such
 sequence contains  m  random elements  𝐶𝑖𝑗  chosen uniformly from the Clifford
 group on  n-qubits, and the  m+1  element is the inversion defined as follows: 
 𝐶_𝑖_𝑚+1=(𝐶_𝑖_1⋅...⋅𝐶_𝑖_𝑚)^(−1) . It can be found efficiently by the Gottesmann-Knill 
theorem.


def randomized_benchmarking_seq(nseeds=1, length_vector=None,
                                rb_pattern=None,
                                length_multiplier=1, seed_offset=0,
                                align_cliffs=False,
                                interleaved_gates=None,
                                is_purity=False):

length_vector: 'm' length vector of Clifford lengths. Must be in
        ascending order. RB sequences of increasing length grow on top of the
        previous sequences.

rb_pattern: A list of the form [[i,j],[k],...] which will make
        simultaneous RB sequences where
        Qi,Qj are a 2Q RB sequence and Qk is a 1Q sequence, etc.
        E.g. [[0,3],[2],[1]] would create RB sequences that are 2Q for Q0/Q3,
        1Q for Q1+Q2
        The number of qubits is the sum of the entries.
        For 'regular' RB the qubit_pattern is just [[0]],[[0,1]].

For example, we generate below several sequences of 2-qubit Clifford circuits.
'''
#Generate RB circuits (2Q RB)
rb_opts = {}
#Number of Cliffords in the sequence
rb_opts['length_vector'] = [2, 4, 16, 32]


rb_opts['nseeds'] = 3 #Number of random sequences
rb_opts['seed_offset']=100  # What to start the seeds at (e.g. if we want to add more seeds later)
#Default pattern
#Default pattern : list of qubits to test and the pairs of qubits to test
rb_opts['rb_pattern'] = [[0,1],[2,3]]  # includes cx
#rb_opts['rb_pattern'] = [[0],[2,3]]

#Do three times as many 1Q Cliffords
length_multiplier = [1,3] #it scales each rb_sequence by the multiplier.
rb_opts['length_multiplier'] = length_multiplier

rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)

#As an example, we print the circuit corresponding to the first RB sequence
print(rb_circs[0][0])

print('A-len',len(rb_circs))
for x in rb_circs:
    print('B-len',len(x))
    for circ in x:
        print(circ.name,', depth:',circ.depth(),', size:',circ.size())
    break
print('xdata:',xdata)  #the Clifford lengths


'''
***** STEP 2a ******   Simulate noise
We can execute the RB sequences either using Qiskit Aer Simulator (with some noise model) or using IBMQ provider, and obtain a list of results.
'''

# Run on a noisy simulator
noise_model = NoiseModel()

# Depolarizing error on the gates u2, u3 and cx (assuming the u1 is virtual-Z gate and no error)
p1Q = 0.002
p2Q = 0.01

noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1Q, 1), 'u3')
noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')

backend =Aer.get_backend('qasm_simulator')
''''
******** Step 2b: Execute the RB sequences 
Step 4: Find the averaged sequence fidelity
Step 5: Fit the results

'''


basis_gates = ['u1','u2','u3','cx'] 
shots = 200
qobj_list = []
rb_fit = rb.RBFitter(None, xdata, rb_opts['rb_pattern'])
for rb_seed,rb_circL in enumerate(rb_circs):
    print('Compiling seed %d, circLen=%d'%(rb_seed,len(rb_circL)))
    rb_circT = transpile(rb_circL, basis_gates=basis_gates, optimization_level=1)
    qobj = assemble(rb_circT, shots=shots)
    if rb_seed==0:
        for circ in rb_circT: print('circT:',circ.name,', depth:',circ.depth(),', size:',circ.size())

    if  rb_seed==0: print(rb_circT[0])
                            
    job = backend.run(qobj, noise_model=noise_model, backend_options={'max_parallel_experiments': 0})
    qobj_list.append(qobj)
    # Add data to the fitter, and  recalculate the raw data, means and fit.
    rb_fit.add_data(job.result())
    f0=rb_fit.fit[0]
    print('After seed %d, alpha: %.3f +/- %.3g, EPC: %.3f +/- %.3g'%(rb_seed,f0['params'][1],f0['params_err'][1], f0['epc'], f0['epc_err']))
    print('   A%d=%.3f +/- %.3f  , B0=%.3f +/- %.3f '%(rb_seed,f0['params'][0],f0['params_err'][0],f0['params'][2],f0['params_err'][2]),'num fit=',len(rb_fit.fit),'rb_pat=',rb_opts['rb_pattern'][0])

    # alpha=rb_fit.fit[0]['params'][1]
    # EPC=rb_fit.fit[0]['epc']
    #break

plt.figure(figsize=(8, 6))
ax = plt.subplot(1, 1, 1)

# Plot the essence by calling plot_rb_data
rb_fit.plot_rb_data(0, ax=ax, add_label=True, show_plt=False)
    
# Add title and label

nQ=rb_circs[0][0].num_qubits
ax.set_title('%d Qubit RB'%(nQ), fontsize=18)

'''
Predicted Gate Fidelity
If we know the errors on the underlying gates (the gateset) we can predict the fidelity. First we need to count the number of these gates per Clifford.

Then, the two qubit Clifford gate error function gives the error per 2Q Clifford. It assumes that the error in the underlying gates is depolarizing. 
'''    


#Count the number of single and 2Q gates in the 2Q Cliffords
qubits = rb_opts['rb_pattern'][0]

gates_per_cliff = rb.rb_utils.gates_per_clifford(qobj_list, xdata[0],basis_gates, rb_opts['rb_pattern'][0])
#print('gpf shape:',gates_per_cliff)
print('  dump gpf:',gates_per_cliff)
for x in basis_gates:  # mean is taken over 2 qbits?
    print("Number of %s gates per Clifford: %f"%(x,
            np.mean([gates_per_cliff[0][x],gates_per_cliff[1][x]])))
    
# convert from depolarizing error to epg (1Q)
epg_q0 = {'u1': 0, 'u2': p1Q/2, 'u3': 2 * p1Q/2}
epg_q1 = {'u1': 0, 'u2': p1Q/2, 'u3': 2 * p1Q/2}

# convert from depolarizing error to epg (2Q)
epg_q01 = 3/4 * p2Q

# calculate the predicted epc from underlying gate errors 
pred_epc = rb.rb_utils.calculate_2q_epc(
    gate_per_cliff=gates_per_cliff,
    epg_2q=epg_q01,
    qubit_pair=qubits,
    list_epgs_1q=[epg_q0, epg_q1])

print("Predicted 2Q Error per Clifford: %e (qasm simulator result: %e)" % (pred_epc, rb_fit.fit[0]['epc']))

plt.show()

