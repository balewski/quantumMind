#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Study correction-matrix based  readout error mitigation. 
Pure simulations.

Update 2022-05
Follow this tutorial

https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html

Noise model: random bit-flip during measurement
'''
import numpy as np
import scipy.linalg as la
from pprint import pprint

import qiskit as qk
# this is noise model: random bit flip
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.utils.mitigation  import (complete_meas_cal,CompleteMeasFitter)

print('qiskit ver=',qk.__qiskit_version__)

#............................
def get_noise(p):
    error_meas = pauli_error([('X',p), ('I', 1 - p)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # error is applied to measurements
    print('meas-noise-model , flip prob=',p)    
    return noise_model

#............................
def apply_readCorr(meas_fitter,results):
    # Get the filter object
    meas_filter = meas_fitter.filter
    # Results with mitigation
    mit_results = meas_filter.apply(results)
    return mit_results 

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    flip_prob=0.01
    noise_model = get_noise(flip_prob)

    print('\n ***** 1 *****  noisy 2-qubit system: show the problem:')
    num_shots=10000
    for state in ['00','01','10','11']:
        qc = qk.QuantumCircuit(2,2)
        if state[0]=='1':
            qc.x(1)
        if state[1]=='1':
            qc.x(0)  
        qc.measure(qc.qregs[0],qc.cregs[0])

        print(state+'=prep --> measure:',
              qk.execute(qc, qk.Aer.get_backend('qasm_simulator'),noise_model=noise_model,shots=num_shots).result().get_counts())

    print(' circuit for the last inp_state:',state)
    print(qc)

    print('\n ***** 2 *****  2-qubit circut with entangling gate')
    qc2 = qk.QuantumCircuit(2,2,name='circ-jan')
    qc2.h(0)
    qc2.cx(0,1)
    #qc2.measure_all()
    qc2.measure(qc2.qregs[0],qc2.cregs[0]) # this measures all cbits
    print(qc2)
    print('noisy:',qk.execute(qc2, qk.Aer.get_backend('qasm_simulator'),noise_model=noise_model,shots=num_shots).result().get_counts())
    print('ideal:',qk.execute(qc2, qk.Aer.get_backend('qasm_simulator'),shots=num_shots).result().get_counts())


    print('\n ***** 4 *****  use qiskit to measure & apply correction matrix')
    # the default is an even more sophisticated method using least squares fitting.
    # let's stick with doing error mitigation for a pair of qubits. 

    qr = qk.QuantumRegister(2, 'q')
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    meas_calibs.append(qc2)  # add extra circuit, so it can be measured w/ SPAM measurement

    for circuit in meas_calibs:
        print('Circuit',circuit.name)
        print(circuit)
        print()

    num_shots=1000
    print(' ***** 4.1 *****  ideal, no noise, fit & print cal_matrix')
    # Execute the calibration circuits without noise
    backend = qk.Aer.get_backend('qasm_simulator')
    job = qk.execute(meas_calibs, backend=backend, shots=num_shots)
    cal_results = job.result()
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    corM=meas_fitter.cal_matrix
    print(corM)
    print(' det cal_matrix=',np.linalg.det(corM))


    flip_prob=0.05
    print('\n ***** 4.1 *****  flip_prob=%f, print cal_matrix'%flip_prob)
    noise_model = get_noise(flip_prob)
    job = qk.execute(meas_calibs, backend=backend, shots=num_shots, noise_model=noise_model)
    cal_results = job.result()
    rawRes=cal_results.get_counts()
    print('raw counts:')
    for cnt,circ in zip(rawRes,meas_calibs):
        print(circ.name,cnt)

    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    print('cal_matrix:');print(meas_fitter.cal_matrix)
    print(' det cal_matrix=',np.linalg.det(meas_fitter.cal_matrix))


    print('\n ***** 4.2 *****  , mitigate its own results')
    mit_results=apply_readCorr(meas_fitter,cal_results)
    pprint(mit_results.get_counts())


    print('\n ***** 4.3 *****  , mitigate 2Q entangled circuit')
    job=qk.execute(qc2, qk.Aer.get_backend('qasm_simulator'),noise_model=noise_model,shots=num_shots)
    results = job.result()
    rawC=results.get_counts()
    print('qc2 raw counts:'); pprint(rawC)
    mit_results=apply_readCorr(meas_fitter,results)
    print('qc2 mit counts:');mitC=mit_results.get_counts(); 
    pprint(mitC)

    s1=0
    for txt in rawC: s1+=rawC[txt] 
    s2=0
    for txt in mitC: s2+=mitC[txt] 
    print('test sums rawC=%.1f mitC=%.1f'%(s1,s2))

