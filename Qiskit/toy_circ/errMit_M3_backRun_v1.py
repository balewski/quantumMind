#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Use Matrix-free Measurement Error Mitigation (M3) w/o smapler()
Instead uses results from backend.run()
Runs on real HW & FakeHanoi

https://github.com/Qiskit-Extensions/mthree

Based on https://qiskit.org/ecosystem/mthree/sampling.html

'''

import time,os,sys
from pprint import pprint
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, transpile
from collections import Counter
from qiskit.providers.fake_provider import FakeHanoi
import mthree  # <=== Matrix-free Measurement Error Mitigation 
from mthree._helpers import system_info as backend_info

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-n','--numShot',type=int,default=5000, help="shots ")
    parser.add_argument('-b','--backend',default='ibm_hanoi',   help="backend for transpiler" )
    parser.add_argument( "-F","--fakeSimu", action='store_true', default=False, help="will switch to backend-matched simulator")
    parser.add_argument( "-J","--jsonM3Calib", action='store_true', default=False, help="will read backend calibration from json")
    args = parser.parse_args()

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    args.outPath='out/'
    assert os.path.exists(args.outPath)

    return args

#...!...!....................
def bv_ones_circs(N):
    """Create a Bernstein–Vazirani (BV) circuit of width N
    Here the bit-string length is N-1
    Parameters:
        N (int): Number of qubits in circuit
    Returns:
        QuantumCircuit: BV circuit
    """
    qc = QuantumCircuit(N, N-1)
    qc.x(N-1)
    qc.h(range(N))
    qc.cx(range(N-1), N-1)
    qc.h(range(N-1))
    qc.barrier()
    qc.measure(range(N-1), range(N-1))
    return qc



#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    #.... prep circuit list
    bit_range = range(3,8)
    qcEL = [bv_ones_circs(N+1) for N in bit_range]
    nCirc=len(qcEL)
    shots = args.numShot
    
    if args.fakeSimu:  # use fake_hanoi noise model
        backend = FakeHanoi()
        print('use fake noisy_backend =',backend.name() )
    else:
        service = QiskitRuntimeService()
        print('M: acquire backend:',args.backend)
        backend = service.get_backend(args.backend)

    backMD=backend_info(backend)
    print('\nmy backend:');pprint(backMD)
    
    print('M: transpile...')
    qcTL = transpile(qcEL, backend=backend, optimization_level=3, seed_transpiler=42)
    jc=0
    qc1=qcTL[jc]
    print(qc1.draw(output='text',idle_wires=False)) 
      

    #.... prime  M3 object ....
    mit = mthree.M3Mitigation(system=backend)
    qMapL = mthree.utils.final_measurement_mapping(qcTL)
    print('M3 mapping: jc=%d %s '%(jc,qMapL[jc]))

    qpuJ='qpu_%s.m3cal.json'%backMD['name']
    qpuF=os.path.join(args.outPath,qpuJ)
    if args.jsonM3Calib:  #
        print('use existing M3Cal:',qpuF)
        mit.cals_from_file(qpuF)
        print('mit.single_qubit_cals:',mit2.single_qubit_cals)
    else: # Collect SPAM data first and save them to a JSON
        shots2=max(1e4,shots)
        print('M: run job to collect SPAM data ..., shots=%d'%shots2,backMD['name'])
        mit.cals_from_system(qubits=qMapL, cals_file=qpuF, shots=shots2) # min(1e4, max_shots)
        print('M: SPAM job done, saved:',qpuF)

    #........ main coircuits .....
    print('M: run problem, nCric=%d on %s  shots=%d'%(nCirc,backMD['name'],shots))    
    countsL = backend.run(qcTL, shots=shots).result().get_counts()
    print('M: payload job done')

    countsCL=[ Counter(data) for data in countsL]
    print('counts:',countsCL[jc])
    raw_success_probs = []
    
    for idx, num_bits in enumerate(bit_range):
        max_bitstring, max_value = countsCL[idx].most_common(1)[0]
        #print('ddd',idx,num_bits,max_bitstring, max_value)
        if max_bitstring != '1'*num_bits: max_value=0
        raw_success_probs.append(max_value/shots)

    print('raw succsss prob:',raw_success_probs)
       
    quasis = mit.apply_correction(countsL, qMapL, return_mitigation_overhead=True)    
    quasisCL=[ Counter(data) for data in quasis]
 
    mit_success_probs = []
    for idx, num_bits in enumerate(bit_range):
        max_bitstring, max_value = quasisCL[idx].most_common(1)[0]
        if max_bitstring != '1'*num_bits:  max_value=0
        mit_success_probs.append(max_value)

    print('mit succsss prob:',mit_success_probs)
    print('mit expval:',quasis.expval()) #  default case of Z 
    
    #....  error analysis
    
    mitOverhead=quasis.mitigation_overhead
    regStd=quasis.stddev()
    print('\n mit overhead=',mitOverhead)
    print('expval_and_stddev:',quasis.expval_and_stddev())

    print('M:done',backMD['name'])
