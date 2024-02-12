#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
  Hadamard Test for 1 qubit
 Input:
  |x> is random 1-qubit state
  U is one of Pauli 3 matrices : x,y,z
 Output: 
    Re (  <x|U|x> )
    Im (  <x|U|x> )

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
   
    parser.add_argument('-U','--Uop', default='x',choices=['x','y','z'], help=" select Pauli operator for U")
 
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

#...!...!..................
def hadamTest_1q(ivec,op,type='re'): # Hadamard-test for 1 qubit initial state and one pauli gate
    qc = QuantumCircuit(2) 
    qc.h(0)
    if type=='im': qc.sdg(0)  # will compute imaginary component
    qc.initialize(ivec, 1) # Apply initialisation operation to the 1st qubit
    if op=='x': qc.cx(0,1)
    elif op=='y': qc.cy(0,1)
    elif op=='z': qc.cz(0,1)    
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
    Uop=args.Uop
    
    #ini_vec = Statevector.from_label('+')    # Fixed  initial_state as |+>
    ini_vec = random_statevector(2)   #  sampled from the uniform (Haar) measure.
    print('ini_vec',ini_vec.data)
    
    sim = Aer.get_backend('aer_simulator')  # Tell Qiskit how to simulate our circuit
 

    qc=evolve_1q(ini_vec,op=Uop)
    qc.save_statevector('ov1')
    qobj = assemble(qc)     # Create a Qobj from the circuit for the simulator to run
    result = sim.run(qobj).result() # Do the simulation and return the result
    ov1 = result.data(0)['ov1']
    print('out vec 1', ov1.data)
    overlap=ini_vec.inner(ov1)
    print('TRUTH1: <svo|sv1>= ',overlap)


    # .... Hadamard-test for real component
    qc2=hadamTest_1q(ini_vec,op=Uop)
    qc2.save_statevector('ov2')
    qobj2 = assemble(qc2)     
    result2 = sim.run(qobj2).result() # Do the simulation and return the result
    ov2 = result2.data(0)['ov2']
    print('out vec 2', ov2.data)
    p0,p1=ov2.probabilities([0])
    print('p0,p1:',p0,p1, 'p0-p1:',p0-p1)
    diff=np.real(overlap)-(p0-p1)
    if abs(diff)<1e-4:
        print('CHECK Real PASSED diff=%.3f  Uop=%s'%(diff,Uop))
    else:
        print('CHECK Real *** FAILED***',diff) ; bad1

    # .... Hadamard-test for real component
    qc3=hadamTest_1q(ini_vec,op=Uop,type='im')
    qc3.save_statevector('ov3')
    qobj3 = assemble(qc3)     
    result3 = sim.run(qobj3).result() # Do the simulation and return the result
    ov3 = result3.data(0)['ov3']
    print('out vec 3', ov3.data)
    p0,p1=ov3.probabilities([0])
    print('p0,p1:',p0,p1, 'p0-p1:',p0-p1)
    diff2=(p0-p1)
    if abs(diff2)<1e-4:
        print('CHECK Imaginary PASSED',diff2)
    else:
        print('CHECK Imaginary *** FAILED***',diff2)  ; bad2      

    print('M:done,   Uop=',Uop)
