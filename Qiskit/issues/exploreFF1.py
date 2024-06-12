#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
feed-forward w/ on-cpu computation?


'''

import qiskit
import qiskit_ibm_runtime
import qiskit_aer
print('Qiskit_ibm_runtime version:',qiskit_ibm_runtime.__version__)
print('Qiskit version:',qiskit.__version__)
print('Qiskit Aer version:',qiskit_aer.__version__)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer

from time import time, sleep

import numpy as np
import os
import argparse

def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-n','--num_shot',type=int,default=1, help="shots, aka sequence length")
    parser.add_argument('-x','--inputs', default=[-0.8, -0.9,0.2], nargs=3, type=float, help=' input values in [-1,1]')
    
    args = parser.parse_args()
       
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))   
    return args

#...!...!....................
def and_consecutive_bits(binary_string):
    result = ""
    n = len(binary_string)
    
    for i in range(n - 1):
        # Perform AND operation between consecutive bits
        and_result = int(binary_string[i]) & int(binary_string[i + 1])
        # Append the result to the result string
        result += str(and_result)
    print('and_bits',binary_string, result)
    return result


#...!...!....................
def circA(args):
    thV=np.arccos(args.inputs)
   
    ni=thV.shape[0]
    no=ni-1
    
    qr = QuantumRegister(ni+no)
    cri = ClassicalRegister(ni)
    cro = ClassicalRegister(no)
    qc = QuantumCircuit(qr,cri,cro)
    
    for i in range(ni):
        qc.ry(thV[i],qr[i])
        qc.measure(i,i)
    return qc

#...!...!....................
def circ_addFF(qc,bitsAB):
    bitA,bitsB=bitsAB.split(' ')
    #print('bbb',bitA,bitsB)
    # Expand circuit by adding feed-forward gate
    L1 = and_consecutive_bits(bitsB)
    no=len(L1)
    ni=no+1
    print('L1=',L1,no,ni)
    qc.barrier()

    for i in range(no):
        iq=ni+i
        if L1[i] == '1':
            qc.x(iq)
    for i in range(no):
        iq=ni+i
        qc.measure(iq,iq)

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    np.set_printoptions(precision=4)
    args=commandline_parser()
    shots=args.num_shot

    backend = AerSimulator(method='automatic')
    print('\n circ A')
    qc=circA(args)
    print(qc)

    #..... measure & print
    job = backend.run(qc, memory = True, shots=shots)
    result = job.result()
    measL=result.get_memory()
    print('shots=%d, measL:'%shots,measL)

    #.... add feed-forward
    circ_addFF(qc,measL[0])
    print(circuit_drawer(qc, output='text',cregbundle=False))

    #..... measure & print
    job = backend.run(qc, memory = True, shots=shots)
    result = job.result()
    measL=result.get_memory()
    print('shots=%d, measL:'%shots,measL)

    
    print('ok')

    
