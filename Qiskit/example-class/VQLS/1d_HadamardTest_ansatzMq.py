#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
 Hadamard Test using  VQLS ansatz for multi qubit shot-based analysis.
 can use FakeLime backend
 VQLS ansatz is optimal for linear connectivity 
 Input:
  |psi> is random m-qubit state  defined by   VQLS ansatz 
  use hardware-efficient layered ansatz from https://arxiv.org/pdf/1909.05820.pdf

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

from qiskit.circuit import ParameterVector
# For simulation
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeLima

import numpy as np
import itertools

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument('-n','--numShots',type=int,default=30000, help="shots")
    parser.add_argument('-M','--numQubits', default=4, type=int, help=" the size of psi state")
    parser.add_argument('-L','--numLayers', default=4, type=int, help="Ansatz consists of L layers: Ry+CZ")
    parser.add_argument( "-F","--fakeIbmq", action='store_true', default=False, help="use fake IBMQ backend")

    args = parser.parse_args()
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
    assert args.numQubits%2==0
    return args

#...!...!..................
def evolve_Mq(qc0,ops): # evolves 1 qubit initial state by applying one pauli gate
    m=qc0.num_qubits
    #print('m',m)
    qc = qc0.copy()
    qc.save_statevector('iv1')
    assert len(ops)==m
    for i,op in enumerate(ops):
        if op=='x': qc.x(i)
        if op=='y': qc.y(i)
        if op=='z': qc.z(i)            
    if verb: print(qc)
    return qc

#...!...!..................
def hadamTest_Mq(qc0,ops): # Hadamard-test for m qubit initial state and  pauli^m gates
    m=qc0.num_qubits
    #print('m',m)
 
    qc = QuantumCircuit(m+1,m+1)  # add ancilla as qubit 0
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
def one_pair_test_shots(qcIni,Uops,shots): # ground truth + Hadamard-test for  one Uops

    # ... ground truth
    qc=evolve_Mq(qcIni,ops=Uops)
    qc.save_statevector('ov1')
    result = backend1.run(qc).result() # Do the simulation and return the result
    iv1 = result.data(0)['iv1']
    ov1 = result.data(0)['ov1']
    if verb>1: print('out vec 1', ov1)
    overlap=iv1.inner(ov1)
    if verb>0: print('TRUTH1: <svo|sv1>= ',overlap)

    # .... Hadamard-test for real component
    qc2=hadamTest_Mq(qcIni,ops=Uops)
    qc2.measure([0],[0])
    #print(qc2)
    result2 = backend2.run(qc2,shots=shots).result() # Do the simulation and return the result
    counts = result2.get_counts(0)
    #print(counts)
    
    lab1=''.join(['0' for i in range(args.numQubits) ])+'1'
    n1=counts[lab1] 
    p1=n1/shots
    numer=n1* (shots -n1)
    if numer==0: numer =1
    p1Err= np.sqrt( numer/shots)/shots
    #print('meas p1=%.3f +/- %.3f'%(p1,p1Err))
    sigz=1.- 2*p1
    sigzErr=2*p1Err
    diff=np.real(overlap)-sigz
    if  args.fakeIbmq :
        isOK=abs(diff)<0.05  # absolute tolerance
    else:
        isOK=abs(diff)<sigzErr*4  # relative tolerance for missing the ground truth
    
    if isOK:
        print('CHECK Real PASSED diff=%.3f err=%.3f Uops=%s shots=%d  truth=%.4f'%(diff,sigzErr,Uops,shots,np.real(overlap)))
    else:
        print('CHECK Real ***FAILED*** diff=%.3f err=%.3f Uops=%s shots=%d  truth=%.4f'%(diff,sigzErr,Uops,shots,np.real(overlap)))
    return isOK

#...!...!..................
def parametrized_linear_ansatz(nq,nlay):

    qc = QuantumCircuit(nq)
    parLL=[]
    nPar=0
    for lay in range(nlay):
        i0=lay%2
        m=nq-2*i0
        #print('lay',lay,i0,m,nlay)        
        parV=ParameterVector('p%d'%lay, m)
        for i in range(m): 
            qc.ry(parV[i], i+i0)
        parLL.append(parV)
        nPar+=m

        if lay==nlay-1: break  # skip last CZs
        for i in range(i0,nq-i0,2):
            #print(' q',i)
            qc.cz(i,i+1)
        #
    print(parLL)
    print(qc)
    return qc,parLL,nPar

#...!...!..................
def instantiate_circuits(parLL,parVal,qc):
    my_dict = {}
    i0=0
    for parL in parLL:
        print('play',i0,parL)
        for i,par in enumerate(parL):
            my_dict[par]=parVal[i0+i]
        i0+=i
    return qc.bind_parameters(my_dict)  
    
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    nq=args.numQubits
    verb=args.verb
    qcPar,parLL,nPar=parametrized_linear_ansatz(nq,args.numLayers)

    parVal=np.random.rand(nPar)*np.pi/4
    qcIni=instantiate_circuits(parLL,parVal,qcPar)
    print(qcIni)
    

    backend1 = Aer.get_backend('aer_simulator')  # Tell Qiskit how to simulate our circuit
    backend2 = AerSimulator.from_backend(FakeLima()) if args.fakeIbmq else backend1
    
    # generate list of all possible nq qubit U operator
    inpLL=[['x','y','z']]*nq
    UopsL=[ele for ele in itertools.product(*inpLL)]
    numU=len(UopsL)
    print('M: Uops %d list'%numU)
    if nq<3: print(UopsL)
    
    nOK=0
    for U in UopsL:
        nOK+=one_pair_test_shots(qcIni,U,args.numShots)
        if nq>1:  verb=0  # will print a lot only once
  
    print('\nM:done,  %d of %d  Uops passed using %d shots on %s'%(nOK,numU,args.numShots,backend2))
