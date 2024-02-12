#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
verify truth table for RGQFTMultiplier from Qiskit

'''
import time,os,sys
from pprint import pprint

import numpy as np
import qiskit as qk
from qiskit import Aer, QuantumCircuit, transpile,execute
from qiskit.circuit.library import RGQFTMultiplier

from bitstring import BitArray
class MyEmpty: pass

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-b','--numDataBits',type=int,default=2, help="number of bits encoding integer")
    parser.add_argument('-n','--numShots',type=int,default=1000, help="shots")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")
    parser.add_argument( "-B","--noBarrier", action='store_true', default=False, help="remove all bariers from the circuit ")

 
    args = parser.parse_args()
    args.randomSeed=123
    args.noise=False
    print('qiskit ver=',qk.__qiskit_version__)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def circ_depth_aziz(qc,text='myCirc'):   # from Aziz @ IBMQ
    len1=qc.depth(filter_function=lambda x: x.operation.name == 'cx')
    len2=qc.depth(filter_function=lambda x: x.operation.num_qubits == 2 )
    len3=qc.depth(filter_function=lambda x: x.operation.num_qubits ==3 )
    len4=qc.depth(filter_function=lambda x: x.operation.num_qubits > 3 )
    print('%s depth: cx-cycle=%d  2c_cycle=%d 3c_cycle=%d 4+_cycle=%d '%(text,len1,len2,len3,len4))

#...!...!....................
def config_me():
    assert args.numDataBits>=2 # too simple for this code
    # pack dims
    cf=MyEmpty()
    cf.nqA=args.numDataBits
    cf.nqTot=4*cf.nqA
   
    print('nqA=%d nqTot=%d'%( cf.nqA,cf.nqTot))
    return cf

#...!...!....................
def circuit_begin(cf,a,b):
    ncR=2*cf.nqA
    qc = QuantumCircuit(cf.nqTot,ncR)

    ab=[a,b]
    for j in range(2): # loop over 2 input registers a,b
        x=ab[j]
        # encode X , binary
        X=BitArray(uint=x,length=cf.nqA)
        X.reverse() # now LSBF
        if args.verb>1: print('j=%d x=%d, lsbf: x=%d bin=%s'%(i,x,X.uint,X.bin))
        for i,v in enumerate(X):
            if v :  qc.x(i+j*cf.nqA)

    if not args.noBarrier: qc.barrier()
    return qc

#...!...!....................
def add_QFTmult(qc,cf):
    # ... append MUL
    qftMUL = RGQFTMultiplier(num_state_qubits=cf.nqA,name='QFTmult')
    qc.compose(qftMUL, inplace=True)
    
    nqAB=2*cf.nqA
    # superposition of A,B + measure
    for i in range(nqAB):
        qc.measure(i+nqAB,i)
   

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    backend=Aer.get_backend("qasm_simulator")    
    cf=config_me()

    if 0:
        qc=circuit_begin(cf,2,3)
        add_QFTmult(qc,cf)
        print(qc)

        basis_gates = ['p','u3','cx']

        print('\nM: QFTmult circuit nqA=%d'%(cf.nqA))
        circ_depth_aziz(qc,'ideal')
        qcT = transpile(qc, backend=backend,basis_gates =basis_gates,  optimization_level=3)
        circ_depth_aziz(qcT,'transp')
        exit(0)
        
    if not args.executeCircuit:
        print('\nNO execution of Qiskit circuit, use -E to run Aer simulator\n')
        exit(0)
    
    maxVal=2**(cf.nqA)
    
    #.....  loop over all possible inputs
    j=0
    for a in range(maxVal):
        for b in range(maxVal):
            j+=1
            print('check j=%d A=%d  B=%d'%(j,a,b))
            qc=circuit_begin(cf,a,b)
            add_QFTmult(qc,cf)
            if cf.nqA==2: print(qc)
            if cf.nqA==3 and a==2 and b==3 : print(qc)
    
            T0=time.time()
            job = execute(qc, backend,shots=args.numShots)
            counts=job.result().get_counts()
            T1=time.time()
            print('M:QCAM %s num keys:'%backend, len(counts),'elaT=%.2f min'%((T1-T0)/60.));  pprint(counts)
            ab=a*b
            X=BitArray(uint=a*b,length=2*cf.nqA)
            tgtStr=X.bin
            print('target: %d*%d=%d, ab.bin:'%(a,b,ab),X.bin,tgtStr)
            assert counts[tgtStr]==args.numShots
        

    print('\nM:end PASSED')
    


