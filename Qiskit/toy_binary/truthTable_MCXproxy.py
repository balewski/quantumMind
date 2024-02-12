#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
The 2 steps decompsition of MCX  by (i) reducing it to multiple CCX and (i) replacing each CCX with 4 CX+ Z(pi/4)  would result with CX-depth of 20 (only) , even for MCX with M=8.

Usage:
 --numControl defines nuber of controls for MCX
(optional)  --doBRE   decomposes CCX → 4*CX+Z(pi/4)
(optional) -E  runs loop over all possible inputs and tests output with 100 shots for each.


'''
import time,os,sys
from pprint import pprint

import numpy as np
import qiskit as qk
from qiskit import Aer, QuantumCircuit, transpile,execute
from bitstring import BitArray

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-k','--numControl',type=int,default=4, help="number of controll lines for MCX")
    parser.add_argument('-n','--numShots',type=int,default=100, help="shots")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")
    parser.add_argument( "-B","--doBRE", action='store_true', default=False, help="Binary Representation proxy for Toffoli --> 4*CX")

 
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

#............................
#............................
#............................
# define my own class overwriteing cxx gate which works for binary encodings
#  Binary Representation Equivalent for Toffoli --> CX
class QuantumCircuit_BRE(QuantumCircuit):
  def __init__(self,nqbit,ncbit=None):
      if ncbit==None:
          QuantumCircuit.__init__(self,nqbit)
      else:
          QuantumCircuit.__init__(self,nqbit,ncbit)
  def ccx(self,i,j,k):
    self.x(i)
    self.x(j)
        
    for m in range(2):
      self.ry(np.pi/4,k)
      self.cx(j,k)
            
      self.ry(np.pi/4,k)
      self.cx(i,k)

    self.x(i)
    self.x(j)
    self.z(k)  # fix common phase

#...!...!....................
def mcx_extend(nq):
    na=nq-3
    print('mcx_extend: nq=%d, nanc=%d'%(nq,na))
    if args.doBRE:
        qc0 = QuantumCircuit_BRE(nq+na)
        qcx = QuantumCircuit_BRE(nq+na)
    else:
        qc0 = QuantumCircuit(nq+na)
        qcx = QuantumCircuit(nq+na)
        
    if nq-1==2: 
         a=0 # nothing to do
        
    if nq-1==3:
        for i in range(na):  qcx.ccx(i*2,i*2+1,nq+i)

    if nq-1==4:
        for i in range(na):  qcx.ccx(i*2,i*2+1,nq+i)

    if nq-1==5:
        for i in range(na-1):  qcx.ccx(i*2,i*2+1,nq+i)
        qcx.ccx(nq-2, nq,nq+na-1)

    if nq-1==6:
        for i in range(na-1):  qcx.ccx(i*2,i*2+1,nq+i)
        qcx.ccx(nq, nq+1,nq+na-1)

    if nq-1==7:
        for i in range(na-2):  qcx.ccx(i*2,i*2+1,nq+i)
        qcx.ccx(nq-2, nq,nq+na-2)
        qcx.ccx(nq+1, nq+2,nq+na-1)

    if nq-1==8:
        for i in range(na-2):  qcx.ccx(i*2,i*2+1,nq+i)
        for i in range(2): qcx.ccx(nq+2*i, nq+2*i+1,nq+4+i)        
    if nq-1==9:
        for i in range(na-3):  qcx.ccx(i*2,i*2+1,nq+i)
        qcx.ccx(nq-2, nq,nq+4)
        for i in range(2): qcx.ccx(nq+2*i+1, nq+2*i+2,nq+5+i)        
        
    #qcx.barrier()  #tmp
    return qcx,qc0
        
#...!...!....................
def mcx_proxy(nq):
    qcx,qc0=mcx_extend(nq)
    qc0.compose(qcx, inplace=True)
    if nq>4:
        s1,s2=2*nq-5,2*nq-4
    elif nq==4:
        s1,s2=2,4
    elif nq==3:
        s1,s2=0,1
    else:
        bad_choice
    print('ss',s1,s2,nq-1)
    qc0.ccx(s1,s2,nq-1)
    qc0.compose(qcx.inverse(), inplace=True)
    return qc0



#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    nq=args.numControl+1

    assert nq in [3,4,5,6,7,8,9,10] # hardcoded solution in  mcx_extend(nq)
    qc1 = QuantumCircuit(nq)
    ctrlq=[i for i in range(nq-1)]
    qc1.mcx(ctrlq,nq-1)
    print(qc1)
    
    nanc=nq-3
    qc2=mcx_proxy(nq)
    print(qc2)

    backend=Aer.get_backend("qasm_simulator")    
    basis_gates = ['p','u3','cx']

    print('\nM: MCX circuit M=%d'%(nq-1))
    circ_depth_aziz(qc1,'MCX ideal')
    qc1T = transpile(qc1, backend=backend,basis_gates =basis_gates,  optimization_level=3)
    circ_depth_aziz(qc1T,'MCX transp')

    print('\nM: CCX+anc circuit M=%d'%(nq-1))
    circ_depth_aziz(qc2,'CCX+anc ideal')
    qc2T = transpile(qc2, backend=backend,basis_gates =basis_gates,  optimization_level=3)
    circ_depth_aziz(qc2T,'CCX+anc transp')

    if not args.executeCircuit:
        print('\nNO execution of Qiskit circuit, use -E to run Aer simulator\n')
        exit(0)
    
    maxVal=2**(nq)
    ctrTag='1'*(nq-1)

    #.....  loop over all possible inputs
    for j in range(maxVal):
        A=BitArray(uint=j,length=nq)
        B=A.copy()

        if ctrTag==A[1:].bin : # trigger  MSB  flip
            B[0]=not B[0]
 
        #print('j=%d'%j, A.bin, ctrTag,B.bin)
        
        A.reverse() # now we have LSBF for setting the state
    
        print('check j=%d A=%s  B=%s'%(j,A.bin,B.bin))
        qc=QuantumCircuit(nq+nanc,nq)
        for i in range(nq):
            if A[i] : qc.x(i)

        qc.compose(qc2, inplace=True)
        qc.barrier()
        for i in range(nq): qc.measure(i,i)
        print(qc) #; mm

        T0=time.time()
        job = execute(qc, backend,shots=args.numShots)
        counts=job.result().get_counts()
        T1=time.time()
        print('M:QCAM %s num keys:'%backend, len(counts),'elaT=%.2f min'%((T1-T0)/60.));  pprint(counts)
        tgtStr=B.bin
        print('target:',B.bin,tgtStr)
        assert counts[tgtStr]==args.numShots
        

    print('\nM:end PASSED')
    


