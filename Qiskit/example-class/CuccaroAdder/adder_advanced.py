#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Advanced Cuccaro adder  for n=1,2,3... bit-wide operands
Uses consecutive qubits for operands  
MSBF data bit assigned to highest qubit
adder circuit separated form initial state circuit
allows for Binary Representation Equivalent for Toffoli --> CX

note: _v2   uses september convention for  qubits-clbits mapping: 
qubits 0,1,...,N-1 will map to the bits of the integer representation : b_0, b_1, …, b_(N-1)  , where 0 is LSB, and N-1 is MSB

 for Cuccaro adder there are 2 operands: A,B and address.
 Operand A uses  lower ids qubits, operand B uses higher ids qubits. 
the 2 ancilla qubits are the MSB bits of output A,B
Diagram for sum:  (0,A,0,B) → (0,A, B+A); for diff : (0,A,B*) → (A, B*-A), where '*' denotes additional MSB set on input 


Implemented based on paper, but heavily modiffied
https://arxiv.org/pdf/quant-ph/0410184.pdf
"A new quantum ripple-carry addition circuit", Steven A. Cuccaro at all


'''
import time,os,sys
from pprint import pprint

import numpy as np
import qiskit as qk
from qiskit.tools.monitor import job_monitor
from qiskit import Aer, execute
from qiskit.converters import circuit_to_dag

from CuccaroAdder_v2 import  ( CuccaroAdder_v2 as  CuccaroAdder,  eval_adder_v2   as eval_adder )

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],  help="increase output verbosity", default=1, dest='verb')
   
    parser.add_argument( "-G","--doGradient", action='store_true', default=False, help=" changes addet to gradient")
    parser.add_argument( "-B","--doBRE", action='store_true', default=False, help="Binary Representation Equivalent for Toffoli --> 4*CX")
    
    parser.add_argument('-b','--numBits',type=int,default=2, help="size of each operand")
 
    args = parser.parse_args()
    args.numShots=100
    print('qiskit ver=',qk.__qiskit_version__)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def circuit_cxDepth_info(qc,verb=1):
    basis_gates=['u3','cx']
    print('vv',verb)
    if verb>0:print('Analyze transpiled CX cycles of %d qubit circ in basis:'%qc.num_qubits,basis_gates)
    qcT= qk.transpile(qc, basis_gates=basis_gates,optimization_level=2)
    if verb>1: print('transp:\n',qcT)
    qdag = circuit_to_dag(qcT)
    dagLay=qdag.layers()

    cnt={ k:np.zeros(qc.num_qubits,dtype=np.int16) for k in ['u3','cx','any'] }
    
    for k,lay in enumerate(dagLay):
        m1=0; m2=0 # counts 1,2 qubit gates in this layer
        for op in lay['graph'].op_nodes():
            if 'u3'==op.name: m1+=1
            if 'cx'==op.name: m2+=1
        m=m1+m2  # total number of opes per cycle
                
        if m1: cnt['u3'][m1-1]+=1                
        if m2: cnt['cx'][m2-1]+=1
        if m:  cnt['any'][m-1]+=1
        
        #print('lay=%d m1=%d m2=%d'%(k,m1,m2))
        
    if verb>0:
        print(' cycle  histo, indexed from 1 :',cnt)
        print(' cycle summary:')
        for x in cnt: print('%s-cycles: %d'%(x,np.sum(cnt[x])))
    if verb>1:
        circMD=qdag.properties()
        pprint(circMD)

  
    
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    backend=Aer.get_backend("qasm_simulator")
    adder=CuccaroAdder(args)
    #print(adder.circ)
    
    circuit_cxDepth_info(adder.circ)
  
    
    
    numx=2**args.numBits    
    nInp=numx**2  # there are 2 n-bit inputs
    iRL=[i for i in range(nInp)]

    #iRL=[2]  # if you want just one pair of inputs
    
    jobD={} ; inpD={}
    print('\nExecute %d **Adder** jobs '%(len(iRL)))

    for iR in iRL: # assemble jobs to be submitted for execution
        xI=iR//numx
        yI=iR%numx
        circ=adder.iniState(xI,yI).compose(adder.circ)
        circ.measure_all(add_bits=False) # measurements will instead be stored in the already existing classical bits
        
        
        inpD[iR]=[xI,yI]
        if iR in [13] :
            print(iR,'inp_x,y/bin:',xI,yI);print(circ)        
            print('. . .   measure  for x=%d y=%d on %s iR=%d'%(xI,yI,backend,iR))
        jobD[iR]= qk.execute(circ, backend, shots=args.numShots,seed=123)
        
        #break  # just 1 pass of this loop

    outInfo={'backend':str(backend),'shots':args.numShots,'nbit':args.numBits}
   
    eval_adder(jobD,adder.meta,inpD)
    
    print('M:end')
    


