#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
the Local readout mitigator in qiskit-terra.
Study local (aka tensorized) correction-matrix based  readout error mitigation. 
Pure simulations.

Update 2022-06
Follow this tutorial

https://qiskit.org/documentation/experiments/tutorials/readout_mitigation.html

Noise model: fake_paris
'''
import numpy as np
from pprint import pprint

import qiskit as qk
print('qiskit ver=',qk.__qiskit_version__)

from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeLima

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.utils.mitigation import tensored_meas_cal, TensoredMeasFitter

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":

    shots=5000
    
    # Generate the calibration circuits
    qr = qk.QuantumRegister(5)
    # we assume: error acts locally on qubit 2 and the pair of qubits 3 and 4.
    
    mit_pattern = [[2],[3,4]]  # LSB order on this list
    #mit_pattern = [[2,1,3,4]]
    meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')

    nqubl=len(mit_pattern)  # qubits blocks
    print('M: qubits: %s --> %s MIT circuits'%(str(mit_pattern),len(meas_calibs)))
    for circ in meas_calibs:
        print(circ.name)
    print(circ)

    backend = AerSimulator.from_backend(FakeLima())

    # Execute the calibration circuits
    job = qk.execute(meas_calibs, backend=backend, shots=shots)
    cal_results = job.result()
    print('M: results');pprint(cal_results.get_counts())


    meas_fitter = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)
    # The fitter provides two calibration matrices. One matrix is for qubit 2, and the other matrix is for qubits 3 and 4.

    print('\nM: SPAM fitted')
    for i in range(nqubl):
        qubl=mit_pattern[i]
        cmat=meas_fitter.cal_matrices[i]
        fid=meas_fitter.readout_fidelity(i)
        print('\nM:qubits:%s fidelity %.3f, cmat:\n'%(str(qubl),fid),cmat)

    # define & execute circuit of iterest
    # Make a 3Q GHZ state - qubits must match
    cr = ClassicalRegister(3)
    ghz = QuantumCircuit(qr, cr)
    ghz.h(qr[2])
    ghz.cx(qr[2], qr[3])
    ghz.cx(qr[3], qr[4])
    ghz.measure(qr[2],cr[0])
    ghz.measure(qr[3],cr[1])
    ghz.measure(qr[4],cr[2])
    
    print('\nMeasure GHZ... on ',backend)
    job = qk.execute([ghz], backend=backend, shots=shots)
    results = job.result()

    # Results without mitigation
    raw_counts = results.get_counts()
    print('GHZ raw counts:'); pprint(raw_counts)
    # Get the filter object
    meas_filter = meas_fitter.filter
    
    print('\nGHZ  results with mitigation on',backend)
    mit_results = meas_filter.apply(results)
    mit_counts = mit_results.get_counts(0)
    for bs in mit_counts:
        print('bs=%s raw=%d mit=%.1f'%(bs,raw_counts[bs],mit_counts[bs]))
    s1=0
    for txt in raw_counts: s1+=raw_counts[txt] 
    s2=0
    for txt in mit_counts: s2+=mit_counts[txt] 
    print('test sums rawC=%.1f mitC=%.1f'%(s1,s2))
        
    #.....................................
    # get ideal results
    backend=Aer.get_backend('qasm_simulator')
    print('\nMeasure GHZ... on ',backend)
    job = qk.execute([ghz], backend=backend, shots=shots)
    results = job.result()

    # Results without mitigation
    raw_counts = results.get_counts()
    print('GHZ raw counts:'); pprint(raw_counts)
        
    print('M:done')    

    '''
    class TensoredFilter()
    def __init__(self,
                 cal_matrices: np.matrix,
                 substate_labels_list: list,
                 mit_pattern: list):
    '''
