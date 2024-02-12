#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
the Local readout mitigator in qiskit-terra.
Study local (aka tensorized) correction-matrix based  readout error mitigation. 
Pure simulations.

Simplify this example to work w/ 3 qubits,  export SPAM meas as dict, create new SPAM fitter from dict, apply it to data, proof the same result is obtained


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
from qiskit.providers.fake_provider import FakeLima

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.utils.mitigation import tensored_meas_cal, TensoredMeasFitter

import sys,os
sys.path.append(os.path.abspath("../utils/"))
from Util_IOfunc import write_yaml, read_yaml
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
    
    mit_pattern = [[4],[2],[3]]  # list all phys qubits used by GHZ circ, order does not matter
    
    circs_spam, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')

    nqubl=len(mit_pattern)  # qubits blocks
    print('M: qubits: %s --> %s MIT circuits'%(str(mit_pattern),len(circs_spam)))
    for circ in circs_spam:
        print(circ.name)
    print(circ)

    backend = AerSimulator.from_backend(FakeLima())

    # Execute the calibration circuits
    job = qk.execute(circs_spam, backend=backend, shots=shots)
    spam_results = job.result()
    print('M: results for all %d SPAM circs'%(len(circs_spam)));print(spam_results.get_counts())

    #print('aa',type(spam_results))  # <class 'qiskit.result.result.Result'>
    spamResD=spam_results.to_dict()
    print('M: spam res dist for %d keys'%(len(spamResD)))
    #1pprint(spamResD['results'][0])
    outF='out/spam1.yaml'
    write_yaml(spamResD,outF)

    spam_results2=qk.result.result.Result.from_dict(spamResD)
    print('M: results2 from dict');print(spam_results2.get_counts())   
    
    spam_fitter = TensoredMeasFitter(spam_results2, mit_pattern=mit_pattern)
    # The fitter provides one calibration matrix  for qubit 3 qubits

    print('\nM: SPAM fitted')
    for i in range(nqubl):
        qubl=mit_pattern[i]
        cmat=spam_fitter.cal_matrices[i]
        fid=spam_fitter.readout_fidelity(i)
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
    print(ghz)
    circT = qk.transpile(ghz, backend=backend, optimization_level=3, seed_transpiler=111)    
    print(circT)
    
    job = qk.execute(circT, backend=backend, shots=shots)
    results = job.result()

    # Results without mitigation
    raw_counts = results.get_counts()
    print('GHZ raw counts:'); pprint(raw_counts)
    
    print('\nGHZ  results with mitigation on',backend)
    mit_results = spam_fitter.filter.apply(results)
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
    print('GHZ ideal counts:'); pprint(raw_counts)
        
    print('M:done')    

    '''
    class TensoredFilter()
    def __init__(self,
                 cal_matrices: np.matrix,
                 substate_labels_list: list,
                 mit_pattern: list):
    '''
