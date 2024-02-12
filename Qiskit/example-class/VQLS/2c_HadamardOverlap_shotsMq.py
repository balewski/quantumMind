#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
  Hadamard-overlap Test for multi qubit - shots based evaluation
 Input:
   |a> is random m-qubit state
   |b> is another random m-qubit state

  A,B are seqence  Pauli 3 matrices : x,y,z, with identity on all other qubits
  E.g. for M=2 qubits: Av=[xI,Iy], len(Av)=M
  This is used by the  'local cost function'  which is more robust to barren plateaus

  A single execution of Hadamard-overlap computes:  Re (  <a|A|a>*<b|B|b> )

  The main code selects a speciffic order of 3*m Paulis, the same for A & B
  E.g. for m=3
  Av=[ x11,y11,z11, 1x1,1y1,1z1, 11x,11y,11z]
  bv=Av

  next it computes the vector of products, len(Ov)=3*m  , '3' because paulis have 3 falvores
  Ov= [ Re (  <a|Ai|a>*<b|Bi|b> ) ]
   

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
def evolve_Mq(ivec,ops,verb=1): # evolves 1 qubit initial state by applying one pauli gate
    m=len(ivec.dims())
    qc = QuantumCircuit(m)  
    qc.initialize(ivec) # Apply initialisation operation
    assert len(ops)==m
    for i,op in enumerate(ops):
        if op=='x': qc.x(i)
        if op=='y': qc.y(i)
        if op=='z': qc.z(i)
        if op=='1': continue
    if verb: print(qc)
    return qc

#...!...!..................
def hadamOverlap_Mq(ivecL,opsL,verb=1): # Hadamard-test for 2*m qubit initial states and  pauli^m gates
    m=len(ivecL[0].dims())
    if verb: print('m',m, opsL)
    qca = QuantumCircuit(m)  # Create a quantum circuit for initalized a-qubits
    qca.initialize(ivecL[0]) # Apply initialisation operation

    qcb = QuantumCircuit(m)  # Create a quantum circuit for initalized b-qubits
    qcb.initialize(ivecL[1]) # Apply initialisation operation

    qc = QuantumCircuit(2*m+1,1)  # add ancilla as qubit  measurement
    qc.h(0)
    if type=='im': qc.sdg(0)  # will compute imaginary component
    qc.compose(qca,qubits=range(1,m+1),inplace=True)
    qc.compose(qcb,qubits=range(1+m,2*m+1),inplace=True)
    for j in range(2):
        ops=opsL[j]
        assert len(ops)==m
        for i,op in enumerate(ops):
            tgt=i+j*m+1
            if op=='x': qc.cx(0,tgt)
            elif op=='y': qc.cy(0,tgt)
            elif op=='z': qc.cz(0,tgt)
            else:  continue
    qc.h(0)
    if verb: print(qc)
    return qc


#...!...!..................
def one_input_truth(ivec,ops,verb=1): # ground truth 
    qc1=evolve_Mq(ivec,ops=ops,verb=verb)
    qc1.save_statevector('ov1')
    result1 = backend.run(qc1).result() # Do the simulation and return the result
    ov1 = result1.data(0)['ov1']
    if verb: print('out vec 1', ov1.data)
    over1=ivec.inner(ov1)
    if verb: print('TRUTH_one: <iv1|ov1>= ',over1)
    return np.real(over1)



#...!...!..................
def one_pair_test(ivecL,Uops,shots): # ground truth + Hadamard-overlay test for a choice of Uops

    # ... ground truth vector for A|a>
    aAa=[]; verb=1
    for op in Uops:
        x=one_input_truth(ivecL[0],op,verb)
        aAa.append(x)
        verb=0  # print only one time
    
    # ... ground truth vector for B|b>
    bBb=[]; verb=1
    for op in Uops:
        x=one_input_truth(ivecL[1],op,verb)
        bBb.append(x)
        verb=0  # print only one time

    print('aAa',aAa)
    print('bBb',bBb)

    truthV=[ x*y for x,y in zip(aAa,bBb)]
    print('truthV',truthV)


    # .... Hadamard-test for real component
    obsV=[]; verb=1
    for op in Uops:  
        qc2=hadamOverlap_Mq(ivecL,opsL=[op,op],verb=verb) # use the same ops for A, B
        qc2.measure([0],[0])
        
        result2 = backend.run(qc2,shots=shots).result() 
        counts = result2.get_counts(0)
        #print(counts)
        n1=counts['1']
        p1=n1/shots
        numer=n1* (shots -n1)
        if numer==0: numer =1
        p1Err= np.sqrt( numer/shots)/shots
        #print('meas p1=%.3f +/- %.3f'%(p1,p1Err))
        sigz=1.- 2*p1
        sigzErr=2*p1Err
        if verb: print('sigz=%.3f +/- %.3f'%(sigz,sigzErr))
        obsV.append([sigz,sigzErr])
        verb=0
        
    X=np.array(truthV)
    Y=np.array(obsV)

    print('truthV',X)
    print('abABab',Y[:,0])
    print('shot Err',Y[:,1])
    D=(X-Y[:,0])/Y[:,1]
    print('res nSig',D)
    l2=np.linalg.norm(D)
    if abs(l2)<9:  # set threshold at nSig=3
        print('\nCHECK Real PASSED  L2/sig2=%.3f  Uops=%s'%(l2,','.join(Uops)))
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
    
    iveca = random_statevector(2**nq) # 2^M dimensional Hilbert space    
    ivecb = random_statevector(2**nq) # 2^M dimensional Hilbert space    
    print('ini_veca',iveca.data)
    print('ini_vecb',ivecb.data)

    backend = Aer.get_backend('aer_simulator') 

    # generate list of all possible nq qubit U operator
    pauliL=['x','y','z']
    Av=[]
    for i in range(nq):
        b=['1']*nq
        for j in pauliL:
            b[i]=j
            #print(i,j,b)
            Av.append(''.join(b))
            
    print('Av',len(Av),Av)
    one_pair_test([iveca,ivecb],Av,args.numShots)
   
  
    print('\nM:done')
