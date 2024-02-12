#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
 Hadamard-overlap Test for 1 qubit state vector.
 Input:
  |a> is random 1-qubit state
  |b> is another random 1-qubit state
  
  A,B are one of Pauli 3 matrices : x,y,z
 Output: 
    Re (  <a|A|a>*<b|B|b> )

use state vector

'''
import sys,os
from pprint import pprint
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.quantum_info import random_statevector, Statevector
import numpy as np

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
   
    parser.add_argument('-U','--Uops', default='xy', help=" a pair Pauli operator for A,B from x,y,z")
 
    args = parser.parse_args()
     
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
       
    return args

#...!...!..................
def evolve_1q(ivec,op): # evolves 1 qubit initial state by applying one pauli gate
    qc = QuantumCircuit(1)  # Create a quantum circuit with one qubit
    qc.initialize(ivec, 0) # Apply initialisation operation to the 0th qubit
    if op=='x': qc.x(0)
    if op=='y': qc.y(0)
    if op=='z': qc.z(0)    
    print(qc)
    return qc

# Hadamard-overlap for 2 qubita initial state and one pauli gate
#...!...!..................
def hadamOverlap_2q(ivecL,opL,type='re'): 
    qc = QuantumCircuit(3) 
    qc.h(0)
    if type=='im': qc.sdg(0)  # will compute imaginary component
    for i in range(2):
        qc.initialize(ivecL[i], i+1) # Apply initialisation 
        op=opL[i]
        if op=='x': qc.cx(0,i+1)
        elif op=='y': qc.cy(0,i+1)
        elif op=='z': qc.cz(0,i+1)    
        else:  bad_op
    qc.h(0)
    print(qc)
    return qc


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    Uops=args.Uops
    
    #ini_vec = Statevector.from_label('+')    # Fixed  initial_state as |+>
    ini_vec1 = random_statevector(2)   #  sampled from the uniform (Haar) measure.
    ini_vec2 = random_statevector(2)   
    print('ini_vec1:',ini_vec1.data,'\nini_vec2:',ini_vec2.data,'Uops:',Uops )
        
    backend = Aer.get_backend('aer_simulator')  
 
    # do  A|a>
    qc1=evolve_1q(ini_vec1,op=Uops[0])
    qc1.save_statevector('ov1')
    result1 = backend.run(qc1).result() # Do the simulation and return the result
    ov1 = result1.data(0)['ov1']
    print('out vec 1', ov1.data)
    over1=ini_vec1.inner(ov1)
    print('TRUTH_A: <iv1|ov1>= ',over1)

    # do  B|b>
    qc2=evolve_1q(ini_vec2,op=Uops[1])
    qc2.save_statevector('ov2')
    result2 = backend.run(qc2).result() # Do the simulation and return the result
    ov2 = result2.data(0)['ov2']
    print('out vec 2', ov2.data)
    over2=ini_vec2.inner(ov2)
    print('TRUTH_B: <iv2|ov2>= ',over2)

    over12=over1*over2
    print('TRUTH: <iv1|ov1>*<iv2|ov2>= ',over12)

    # .... Hadamard-overlap test for real component
    qc3=hadamOverlap_2q([ini_vec1,ini_vec2],opL=Uops)
    qc3.save_statevector('ov3')
    result3 = backend.run(qc3).result() # Do the simulation and return the result
    ov3 = result3.data(0)['ov3']
    print('out vec3 len:', len(ov3.data))
    p0,p1=ov3.probabilities([0])
    print('p0,p1:',p0,p1, 'p0-p1:',p0-p1)

    diff=np.real(over12)-(p0-p1)
    if abs(diff)<1e-4:
        print('\nCHECK Real PASSED diff=%.3f  Uops=%s  over12=%.3f\n'%(diff,Uops,np.real(over12)))
    else:
        print('\nCHECK Real *** FAILED***',diff)  

    print('M:done,   Uops=',Uops)
