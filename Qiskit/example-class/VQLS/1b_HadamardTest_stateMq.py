#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
 Hadamard Test for multi qubit state vector analysis.
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

    qc = QuantumCircuit(m+1)  # add ancilla as qubit 0
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
def one_pair_test(ini_vec,Uops): # ground truth + Hadamard-test for  one Uops

    # ... ground truth
    qc=evolve_Mq(ini_vec,ops=Uops)
    qc.save_statevector('ov1')
    qobj = assemble(qc)     # Create a Qobj from the circuit for the simulator to run
    result = sim.run(qobj).result() # Do the simulation and return the result
    ov1 = result.data(0)['ov1']
    if verb>1: print('out vec 1', ov1)
    overlap=ini_vec.inner(ov1)
    if verb>0: print('TRUTH1: <svo|sv1>= ',overlap)

    # .... Hadamard-test for real component
    qc2=hadamTest_Mq(ini_vec,ops=Uops)
    
    qc2.save_statevector('ov2')
    qobj2 = assemble(qc2)     
    result2 = sim.run(qobj2).result() # Do the simulation and return the result
    ov2 = result2.data(0)['ov2']
    if verb>1: print('out vec 2', ov2)
    p0,p1=ov2.probabilities([0])
    if verb>1: print('p0,p1:',p0,p1, 'p0-p1:',p0-p1)
    diff=np.real(overlap)-(p0-p1)
    if abs(diff)<1e-4:
        print('CHECK Real PASSED diff=%.3f  Uops=%s'%(diff,Uops))
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

    sim = Aer.get_backend('aer_simulator')  # Tell Qiskit how to simulate our circuit

    # generate list of all possible nq qubit U operator
    inpLL=[['x','y','z']]*nq
    UopsL=[ele for ele in itertools.product(*inpLL)]
    numU=len(UopsL)
    print('M: Uops %d list'%numU,UopsL)
    
    #verb=0
    for U in UopsL:
        one_pair_test(ini_vec,U)
        if nq>1:  verb=0  # will print a lot only once
  
    print('\nM:done,  %d  Uops passed'%numU)
