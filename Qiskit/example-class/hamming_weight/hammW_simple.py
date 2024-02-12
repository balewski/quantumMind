#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Computes hamming weight for 2bit input --> 2 bit output


'''
import time,os,sys
from pprint import pprint

import numpy as np
import qiskit as qk
#from qiskit.tools.monitor import job_monitor
from qiskit import Aer, execute
from bitstring import BitArray
from qiskit import QuantumCircuit
from qiskit.circuit.library  import   CRYGate
from qiskit.converters import circuit_to_dag
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-n','--numShots',type=int,default=100, help="shots")
    parser.add_argument('-q','--numQubits', default=[3,2], type=int,  nargs='+', help='pair: nq_data nq_hammw, space separated ')    
 
    args = parser.parse_args()
   
    print('qiskit ver=',qk.__qiskit_version__)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def init_state(iR):  # prepares initial state
    nq_data,nq_hammw=args.numQubits
    nqtot=nq_data+nq_hammw
    qc = QuantumCircuit(nqtot,nq_hammw)  # LSB first
    
    A=BitArray(uint=iR,length=nq_data)  # MSB first
    print('\nIS:iR',iR,A.bin)
    for i in range(nq_data):
        #print('IS:i',i,A[i])
        if A[i]: qc.x(nq_data-i-1)
    return qc

#...!...!....................
def add_hammw_bit0(qc):
    nq_data,nq_hammw=args.numQubits
    for i in range(nq_data):  qc.cx(i,nq_data)
    qc.barrier()
 

#...!...!....................
def add_hammw_bit1(qc):
    nq_data,nq_hammw=args.numQubits
    qtgt=nq_data+1
    theta=np.pi/2/2 #nq_data
    #qc.x(qtgt-1)
    for i in range(nq_data):  qc.append(CRYGate(theta), [i,qtgt])
    qc.cx(qtgt-1,qtgt)
    for i in range(nq_data):  qc.append(CRYGate(theta), [i,qtgt])
    qc.cx(qtgt-1,qtgt)
    #qc.x(qtgt-1)
    assert nq_data==3
    qc.barrier()

    qc.mcx([0,1,2],qtgt)
    qc.barrier()
    

#...!...!....................
def add_meas(qc):
    nq_data,nq_hammw=args.numQubits
    for i in range(nq_hammw):
        qc.measure(nq_data+i,i)


#...!...!....................
def build_circ(iR):
    qc=init_state(iR)
    add_hammw_bit0(qc)
    add_hammw_bit1(qc)
    add_meas(qc)
    return qc

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

    if 0:  # test one
        qc=init_state(5)
        add_hammw_bit0(qc)
        add_hammw_bit1(qc)
        #add_hammw_bit2(qc)
        add_meas(qc)
        print(qc)
        exit(0)
    
    #ok0
    backend=Aer.get_backend("qasm_simulator")
    basis_gates = ['u3','cx']
    mxval=1<<args.numQubits[0]
    for iR in range(mxval):
        #if iR!=1: continue
        qc=build_circ(iR)
        #print(qc)
        qcT = qk.transpile(qc, backend=backend, basis_gates = basis_gates, optimization_level=2)
        #print(qcT)
        job= backend.run(qcT, backend, shots=args.numShots)
        result = job.result()
        counts = result.get_counts()
        print('   counts=',counts)

    print(qc)
    print(qcT)

    circuit_cxDepth_info(qc)
    print('M:end')
    


