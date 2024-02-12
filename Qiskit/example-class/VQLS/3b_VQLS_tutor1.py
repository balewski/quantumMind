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
#import itertools
from scipy.optimize import minimize

import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument('-n','--numShots',type=int,default=30000, help="shots")
    parser.add_argument('-M','--numQubits', default=2, type=int, help=" the size of psi state")
    parser.add_argument('-L','--numLayers', default=2, type=int, help="Ansatz consists of L layers: Ry+CZ")
    parser.add_argument( "-F","--fakeIbmq", action='store_true', default=False, help="use fake IBMQ backend")

    args = parser.parse_args()
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
    #assert args.numQubits%2==0
    return args


#...!...!..................
def build_VQLS_circ1(qcAns,opsL): #
    m=qcAns.num_qubits
    qc = QuantumCircuit(m+1)  # add ancilla as qubit 0
    qc.h(0)

    qc.compose(qcAns,qubits=range(1,m+1),inplace=True)
    for ops in opsL:
        assert len(ops)==m
        print('kk',ops)
        for i,op in enumerate(ops):
            tgt=i+1
            if op=='x': qc.cx(0,tgt)
            elif op=='y': qc.cy(0,tgt)
            elif op=='z': qc.cz(0,tgt)
            else:  continue
    qc.h(0)
    return qc
    
#...!...!..................
def build_VQLS_circ1(qcAns,opsL): #  only V
    m=qcAns.num_qubits
    qc = QuantumCircuit(m+1)  # add ancilla as qubit 0
    qc.h(0)

    qc.compose(qcAns,qubits=range(1,m+1),inplace=True)
    for ops in opsL:
        assert len(ops)==m
        if verb: print('kk1',ops)
        for i,op in enumerate(ops):
            tgt=i+1
            if op=='x': qc.cx(0,tgt)
            elif op=='y': qc.cy(0,tgt)
            elif op=='z': qc.cz(0,tgt)
            else:  continue
    qc.h(0)
    return qc
    
#...!...!..................
def build_VQLS_circ2(qcAns,qcB,opsL): #  only V
    m=qcAns.num_qubits
    assert len(opsL)==2
    assert len(opsL[0])==m
    qc = QuantumCircuit(2*m+1)  # add ancilla as qubit 0
    qc.h(0)

    qc.compose(qcAns,qubits=range(1,m+1),inplace=True)
    qc.compose(qcB,qubits=range(m+1,2*m+1),inplace=True)
    for j,ops in enumerate(opsL):
        if verb: print('kk2 j ops:',j,ops)
        for i,op in enumerate(ops):
            tgt=i+1+m*j
            if op=='x': qc.cx(0,tgt)
            elif op=='y': qc.cy(0,tgt)
            elif op=='z': qc.cz(0,tgt)
            else:  continue
    qc.h(0)
    return qc
    
#...!...!..................
def parametrized_ansatz(nq,nlay):
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
       
    print(parLL)
    print(qc)
    return qc,parLL,nPar

#...!...!..................
def prep_mHad(m):  # tensor product of m hadamard acting on 0
    qc = QuantumCircuit(m)
    for i in range(m):  qc.h(i)
    print(qc)
    return qc

#...!...!..................
def instantiate_circuits(parLL,parVal,qc):
    my_dict = {}
    i0=0
    for parL in parLL:
        #print('play',i0,parL)
        for i,par in enumerate(parL):
            my_dict[par]=parVal[i0+i]
        i0+=i
    return qc.bind_parameters(my_dict)  

#...!...!..................
def one_QPU_step(parLL,parVal,qcPar):
    qc2=instantiate_circuits(parLL,parVal,qcPar)
    if verb: print(qc2)
    qc2.save_statevector('ov2')
    result2 = backend.run(qc2).result() 
    ov2 = result2.data(0)['ov2']
    if verb>1: print('out vec 2', ov2)
    p0,p1=ov2.probabilities([0])
    delp=p0-p1
    if verb>0: print('p0,p1: %.3f %.3f '%(p0,p1), 'p0-p1: %.3f'%delp)
    return delp

#...!...!..................
def vqls_cost_func(parVal):
    val1=cost_func(parVal,'psi|psi')
    val2=cost_func(parVal,'b|psi')
    loss=1 - val2*val2/val1
    #loss= val2 - val1
    #loss =val2
    print('vqls_cost_func:  psi|psi=%.3f b|psi=%.3f  loss=%.4f'%(val1,val2,loss))
    return loss
          

#...!...!..................
def cost_func(parameters, mode):   #mode :[ 'psi|psi', 'b|psi'] 
    global opt
    sum_1 = 0    
    #parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]
    parVal=parameters
    # gate_set = [[0, 0, 0], [0, 0, 1]]
    nPauli=len(Aset_pauli)
    
    for i in range(0,nPauli):
        for j in range(0, nPauli):
            if verb: print('CFC i,j',i,j)
            AlAl=[  Aset_pauli[i], Aset_pauli[j]]
            if mode=='psi|psi':
                qc=build_VQLS_circ1(qcAnsatz,AlAl)   # only V
            elif mode=='b|psi':
                qc=build_VQLS_circ2(qcAnsatz,qcBstate,AlAl)   # both V and b
            else:
                bad_cost_mode
            p0mp1=one_QPU_step(parLL,parVal,qc)
            cicj=Aset_coef[i]*Aset_coef[j]
            sum_1+=p0mp1*cicj
    if verb>0: print('cost func mode=%s : %.3f\n'%(mode,sum_1))
    return sum_1

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    nq=args.numQubits
    verb=args.verb

    # A = 0.458*Z3 + 0.55*I
    Aset_coef=[0.3, 0.7]
    Aset_pauli=['11z1','1111']
    Bset_gates='hhhh'
 
    qcAnsatz,parLL,nPar=parametrized_ansatz(nq,args.numLayers)
    qcBstate=prep_mHad(nq)
    parVal0=np.random.rand(nPar)*np.pi/4
    backend = Aer.get_backend('aer_simulator')  # Tell Qiskit how to simulate our circuit
    
    if 1: # testing components        
        #qcVQLS=build_VQLS_circ1(qcAnsatz,['1z11','11x1'])  # only V
        #qcVQLS=build_VQLS_circ2(qcAnsatz,qcBstate,['1z11','11x1']) #  both V, b
        #one_QPU_step(parLL,parVal0,qcVQLS)
        #cost_func(parVal0,'psi|psi')
        #cost_func(parVal0,'b|psi')
        vqls_cost_func(parVal0)
        end1

    

    # the main optimizer loop
    numIter=50
    verb=0
    out = minimize(vqls_cost_func, x0=parVal0, 
                   method="COBYLA", options={'maxiter':numIter})
    print('\nM:end of minimizer:\n', out)
