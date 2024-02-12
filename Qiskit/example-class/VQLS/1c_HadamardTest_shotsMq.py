#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
 Hadamard Test for multi qubit shot-based analysis.
 Input:
  |psi> is random m-qubit state
  U is tensor product of all possible m-permutations of  Pauli matrices

  E.g. for M=2 qubits: U={ xx,xy,xz,yx,yy,yz,zx,zy,zz }

 Output: 
    Re (  <psi|U|psi> )
    Im (  <psi|U|psi> )

use state vector

'''
import sys,os
from pprint import pprint
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.quantum_info import random_statevector, Statevector
import numpy as np
import itertools

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument('-n','--numShots',type=int,default=8000, help="shots")
    parser.add_argument('-M','--numQubits', default=2, type=int, help=" the size of psi state")
    args = parser.parse_args()
     
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
       
    return args

#...!...!..................
def evolve_Mq(ivec,ops): # evolves 1 qubit initial state by applying one pauli gate
    m=len(ivec.dims())
    #print('m',m)
    qc = QuantumCircuit(m)  # Create a quantum circuit with one qubit
    qc.initialize(ivec) # Apply initialisation operation to the 0th qubit
    assert len(ops)==m
    for i,op in enumerate(ops):
        if op=='x': qc.x(i)
        if op=='y': qc.y(i)
        if op=='z': qc.z(i)            
    if verb: print(qc)
    return qc

#...!...!..................
def hadamTest_Mq(ivec,ops): # Hadamard-test for m qubit initial state and  pauli^m gates
    m=len(ivec.dims())
    #print('m',m)
    qc0 = QuantumCircuit(m)  # Create a quantum circuit for initalized qubits
    qc0.initialize(ivec) # Apply initialisation operation to the 0th qubit
    #print(qc0)

    qc = QuantumCircuit(m+1,1)  # add ancilla as qubit 0 measurement
    qc.h(0)
    if type=='im': qc.sdg(0)  # will compute imaginary component
    qc.compose(qc0,qubits=range(1,m+1),inplace=True)
    assert len(ops)==m
    for i,op in enumerate(ops):
        if op=='x': qc.cx(0,i+1)
        elif op=='y': qc.cy(0,i+1)
        elif op=='z': qc.cz(0,i+1)
        else:  bad_op
    qc.h(0)
    if verb: print(qc)
    return qc

#...!...!..................
def one_pair_test_shots(ini_vec,Uops,shots): # ground truth + Hadamard-test for  one Uops

    # ... ground truth
    qc=evolve_Mq(ini_vec,ops=Uops)
    qc.save_statevector('ov1')
    result = backend.run(qc).result() # Do the simulation and return the result
    ov1 = result.data(0)['ov1']
    if verb>1: print('out vec 1', ov1)
    overlap=ini_vec.inner(ov1)
    if verb>0: print('TRUTH1: <svo|sv1>= ',overlap)

    # .... Hadamard-test for real component
    qc2=hadamTest_Mq(ini_vec,ops=Uops)
    qc2.measure([0],[0])
    #print(qc2)
   
    result2 = backend.run(qc2,shots=shots).result() # Do the simulation and return the result
    counts = result2.get_counts(0)
    #print(counts)
    #lab1=''.join(['0' for i in range(args.numQubits) ])+'1'
    n1=counts['1'] 
    p1=n1/shots
    numer=n1* (shots -n1)
    if numer==0: numer =1
    p1Err= np.sqrt( numer/shots)/shots
    #print('meas p1=%.3f +/- %.3f'%(p1,p1Err))
    sigz=1.- 2*p1
    sigzErr=2*p1Err
    diff=np.real(overlap)-sigz
    
    if abs(diff)<sigzErr*3:
        print('CHECK Real PASSED diff=%.3f err=%.3f Uops=%s shots=%d  truth=%.4f'%(diff,sigzErr,Uops,shots,np.real(overlap)))
    else:
        print('CHECK Real *** FAILED***',diff) ; bad1


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    nq=args.numQubits
    verb=args.verb
    
    ini_vec = random_statevector(2**nq) # 2^M dimensional Hilbert space    
    print('ini_vec',ini_vec)

    backend = Aer.get_backend('aer_simulator')  # Tell Qiskit how to simulate our circuit

    # generate list of all possible nq qubit U operator
    inpLL=[['x','y','z']]*nq
    UopsL=[ele for ele in itertools.product(*inpLL)]
    numU=len(UopsL)
    print('M: Uops %d list'%numU,UopsL)
    
    #verb=0
    for U in UopsL:
        one_pair_test_shots(ini_vec,U,args.numShots)
        if nq>1:  verb=0  # will print a lot only once
  
    print('\nM:done,  %d  Uops passed'%numU)
