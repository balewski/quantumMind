#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Updated 2022 - this 'rb' lib is not functioning any more, see 8a_ example


Randomized Benchmarking
based on 
https://community.qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html

AND:
https://qiskit.org/documentation/tutorials/noise/4_randomized_benchmarking.html

Can run on a simulator and on the real HW

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
the Clifford group on  1-qubit, including a computed reversal element, 
that should return the qubits to the initial state.

'''
useSimu=True
shots = 200

#Generate RB circuits (1Q RB)
rb_opts = {}
#Number of Cliffords in the sequence
rb_opts['length_vector'] = [1, 10, 20, 50, 75, 100, 125, 150, 175, 200,300,400,500]
#rb_opts['length_vector'] = [1, 2, 4,8]

rb_opts['nseeds'] = 20 #Number of random sequences
rb_opts['rb_pattern'] = [[0]] #Default pattern : list of qubits to test
rb_opts['seed_offset']=33  # What to start the seeds at (e.g. if we want to add more seeds later), Jan: not working

# GENERATE RB CIRCUITS
rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)


# 
print('print all %d circuit corresponding to the 1st RB sequence'%len(rb_circs[0]))
for circ in  rb_circs[0]:
    print(circ)
#print(rb_circs[0][0])

print('A-len, num seeds',len(rb_circs))
for x in rb_circs:
    print(' B-len, num sequences',len(x))
    for circ in x:
        print('  name=',circ.name,', depth:',circ.depth(),', size:',circ.size())
    break
print('xdata, seqs lengths:',xdata)  #the Clifford lengths


if useSimu:
    print(' Run on a noisy simulator')
    noise_model = NoiseModel()

    # Depolarizing error on the gates u2, u3  (assuming the u1 is virtual-Z gate and no error)
    p1Q = 0.003
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1Q, 1), 'u3')

    if 0:
        print('Add T1/T2 noise to the simulation')
        t1 = 50. ;  t2 = 20. ;  gate1Q = 0.1  # duration time for X90, assume unit=uSec
        noise_model.add_all_qubit_quantum_error(thermal_relaxation_error(t1,t2,gate1Q), 'u2')
        noise_model.add_all_qubit_quantum_error(thermal_relaxation_error(t1,t2,2*gate1Q), 'u3')

    
    backend =Aer.get_backend('qasm_simulator')
else:
    numJobs=rb_opts['nseeds']
    print('\n ----------Running circuits from the IBM Q, numJobs=',numJobs)
    if numJobs>5 :
        print('too many jobs, it will take forever, abort')
        #exit(1)
    from qiskit import IBMQ
    from qiskit.tools.monitor import job_monitor
    print('\nIBMQ account:')
    IBMQ.load_account()
    provider = IBMQ.get_provider(group='open')
    #backName='ibmq_ourense' # 5Q, T-topo
    backName='ibmq_santiago' # 5Q, Q1 dead, line-topo
    backend = provider.get_backend(backName)

print('\nmy backend=',backend)
backStat=backend.status().to_dict()
print(backStat)    

''''
******** Step 2b: Execute the RB sequences 
Step 4: Find the averaged sequence fidelity
Step 5: Fit the results

'''

basis_gates = ['u1','u2','u3'] 
#qobj_list = []
transpiled_circs_list = []
rb_fit = rb.RBFitter(None, xdata, rb_opts['rb_pattern'])
for rb_seed,rb_circL in enumerate(rb_circs):
    print('Compiling seed %d, circLen=%d'%(rb_seed,len(rb_circL)))
    new_rb_circ_seed = qiskit.compiler.transpile(rb_circL, basis_gates=basis_gates)
    transpiled_circs_list.append(new_rb_circ_seed)

    if useSimu:
        print('Simulating seed %d'%rb_seed)
        job = qiskit.execute(new_rb_circ_seed, backend, shots=shots,
                         noise_model=noise_model,
                         backend_options={'max_parallel_experiments': 0})    
    else:
        assert backStat['status_msg']=='active'
        print('\nrun on HW job %d of %d'%(rb_seed,numJobs))
        job = qiskit.execute(new_rb_circ_seed, backend, shots=shots)
        jid=job.job_id()
        print('submitted JID=',jid,backend )
        job_monitor(job)


    # Add data to the fitter, and  recalculate the raw data, means and fit.
    rb_fit.add_data(job.result())
    f0=rb_fit.fit[0]
    print('After seed %d, alpha: %.4f +/- %.2g, EPC: %.3g +/- %.2g'%(rb_seed,f0['params'][1],f0['params_err'][1], f0['epc'], f0['epc_err']))
    print('   seed=%d  A=%.3f +/- %.3f  , B=%.3f +/- %.3f '%(rb_seed,f0['params'][0],f0['params_err'][0],f0['params'][2],f0['params_err'][2]),'num fit=',len(rb_fit.fit),'rb_pat=',rb_opts['rb_pattern'][0])


plt.figure(figsize=(8, 6))
ax = plt.subplot(1, 1, 1)

# Plot the essence by calling plot_rb_data
rb_fit.plot_rb_data(0, ax=ax, add_label=True, show_plt=False)
    
# Add title and label
nQ=rb_circs[0][0].num_qubits
ax.set_title('%d Qubit RB, backend=%s'%(nQ,backStat['backend_name']), fontsize=18)

'''
Predicted Gate Fidelity
If we know the errors on the underlying gates (the gateset) we can predict the fidelity. First we need to count the number of these gates per Clifford.

Then, the two qubit Clifford gate error function gives the error per 1Q Clifford. It assumes that the error in the underlying gates is depolarizing. 
'''    


#Count the number of single and 2Q gates in the 2Q Cliffords
qubits = rb_opts['rb_pattern'][0]

gates_per_cliff = rb.rb_utils.gates_per_clifford(transpiled_circs_list, xdata[0],basis_gates, rb_opts['rb_pattern'][0])

#print('  dump gpf:',gates_per_cliff)
for x in basis_gates:
    print("Number of %s gates per Clifford: %f"%(x,gates_per_cliff[0][x]))

if useSimu:
    # convert from depolarizing error to epg (1Q)
    epg_q0 = {'u1': 0, 'u2': p1Q/2, 'u3': 2 * p1Q/2}
    # calculate the predicted epc from underlying gate errors 
    pred_epc = rb.rb_utils.calculate_1q_epc(
        gate_per_cliff=gates_per_cliff,
        epg_1q=epg_q0,
        qubit=0 )
    print("Predicted 1Q Error per Clifford: %e (qasm simulator result: %e)" % (pred_epc, rb_fit.fit[0]['epc']))

plt.show()



