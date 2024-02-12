#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
A bare-bone  Cuccaro adder for n=1,2,3...
Uses interleaved  qubits for operands  ( very user unfriendly)
intial state integrated w/ addedr

Let a=a_2.a_1.a_0;  b=b_2.b_1.b_0
let c=a+b = c_3.c_2.c_1.c_0

Implemented based on paper:
https://arxiv.org/pdf/quant-ph/0410184.pdf
"A new quantum ripple-carry addition circuit", Steven A. Cuccaro at all


'''
import time,os,sys
from pprint import pprint

import numpy as np
import qiskit as qk
from qiskit.tools.monitor import job_monitor
from qiskit import Aer, execute
from CuccaroAdder_v1 import  ( CuccaroAdder_v1 as  CuccaroAdder,  eval_adder_v1  as eval_adder )
from qiskit.converters import circuit_to_dag


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-r","--reorderBits", action='store_true', default=False, help="qubits are reorderd to keep operands bits consecutive")

    parser.add_argument('-n','--numShots',type=int,default=100, help="shots")
    parser.add_argument('-b','--numBits',type=int,default=2, help="size of each operand")
 
    args = parser.parse_args()
    args.randomSeed=123
    args.noise=False
    print('qiskit ver=',qk.__qiskit_version__)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args



#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    backend=Aer.get_backend("qasm_simulator")
    
    
    numx=2**args.numBits    
    nInp=numx**2  # there are 2 n-bit inputs
    iRL=[i for i in range(nInp)]
    
    #iRL=[4] # test one
    
    jobD={} ; circD={}
    print('\nSubmit %d **Adder** jobs '%(len(iRL)))
    iR=0 #; nInp=4
    for iR in iRL: # assemble jobs to be submitted for execution
        #if iR!=nInp//3: continue
        xI=iR//numx
        yI=iR%numx
        circ=CuccaroAdder(args)
        circ.build_circuit(xI,yI)

        circD[iR]=circ
        if iR==nInp//3 :
            print(circ.qc.name,'inp_x,y/bin:',circ.meta['inp_x'].bin,circ.meta['inp_y'].bin);
            print(circ.qc)
            qdag = circuit_to_dag(circ.qc);  circMD=qdag.properties()
            pprint(circMD)
            
            print('. . .   measure sum for x=%d y=%d on %s iR=%d'%(xI,yI,backend,iR))
        jobD[iR]= qk.execute(circ.qc, backend, shots=args.numShots,seed=args.randomSeed)
        
        #break  # just 1 pass of this loop

    outInfo={'backend':str(backend),'shots':args.numShots,'nbit':args.numBits}
    #pprint(circ.meta)
    
    jobCurD={} 
    for name in ['nbit','nqbit']:
            outInfo[name]=circ.meta[name]
    jobCurD['mapABXZ']=circ.mapABXZ
    jobCurD['info']=outInfo

    eval_adder(jobD,circD,jobCurD)
    

    print('M:end')
    


