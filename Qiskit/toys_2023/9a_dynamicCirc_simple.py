#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Explore dynamic circuit: single-qubit 
Runs alwasy on simulator, optional on a real backend
No graphics

My notes: https://docs.google.com/document/d/14JL-yEdeKAxj8HYPUIg-pdARgRmbAuKKLW5FMGzRA2o/edit?usp=sharing

Updated 2023-02
based on :
https://quantum-computing.ibm.com/services/programs/docs/runtime/manage/systems/dynamic-circuits/Introduction-To-Dynamic-Circuits

'''

import time,os,sys
from pprint import pprint

import numpy as np
import qiskit as qk
from packaging import version

from qiskit.tools.monitor import job_monitor
from qiskit_ibm_provider import least_busy
from qiskit_ibm_provider import IBMProvider

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="exec on real backend, may take long time ")
    parser.add_argument('-b','--backName',default='ibmq_kolkata',help="backand for computations, should support dynamic circuits" )
    parser.add_argument('-n','--numShots',type=int,default=2002, help="shots")
    parser.add_argument('-q','--qubit', default=1, type=int, help='physical qubit id')
    parser.add_argument('-L','--optimLevel',type=int,default=1, help="transpiler ")
    #  * 0: no optimization * 1: light optimization * 2: heavy optimization * 3: even heavier optimization If None, level 1 will be chosen as default.
    args = parser.parse_args()

    args.rnd_seed=111 # for transpiler

    print('qiskit ver=',qk.__qiskit_version__)
    assert version.parse(qk.__qiskit_version__["qiskit-terra"]) >= version.parse("0.22")
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def circMXM():
    name='MXM_q%d'%args.qubit
    crPre = qk.ClassicalRegister(1, name="m_pre")
    crPost = qk.ClassicalRegister(1, name="m_post")
    qr = qk.QuantumRegister(1, name="q")
    qc= qk.QuantumCircuit(qr, crPre,crPost,name=name)
    qc.h(0)
    qc.measure(0,crPre)
    qc.x(0)  # unconditionla state flip
    qc.measure(0, crPost)
    return qc
    
#...!...!....................
def circMcondXM():
    name='McondXM_q%d'%args.qubit    
    crPre = qk.ClassicalRegister(1, name="m_pre")
    crPost = qk.ClassicalRegister(1, name="m_post")
    qr = qk.QuantumRegister(1, name="q")
    qc= qk.QuantumCircuit(qr, crPre,crPost,name=name)
    qc.h(0)
    qc.measure(0,crPre)
    with qc.if_test((crPre, 0)): 
        qc.x(0)
    qc.measure(0, crPost)
    return qc
    
#...!...!....................
def verify(qc,counts):    
    
    n1=0; n2=0
    
    if 'MXM' in qc.name:
        k1='0 1';  k2='1 0'
    if 'McondXM' in qc.name:
        k1='1 0';  k2='1 1'

    if k1 in counts: n1=counts.get(k1)
    if k2 in counts: n2=counts.get(k2)
    deln=6*np.sqrt(shots/2)
    print('verify:', qc.name,n1,n2,deln)
    c1= (shots -n1 -n2) <deln
    c2= abs(n1 -n2) <deln
    msg='    PASS    ' if c1 and c2  else '   ***FAILED***'
    print('verify  c1, c2',c1,c2,msg,qc.name,'\n')
    
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    backend_sim = qk.Aer.get_backend('aer_simulator')
    shots=args.numShots
    
    qc1=circMXM()  # X always flip
    qc2=circMcondXM()  # conditional pi-pulse

    qcL=[qc2,qc1]
    job = backend_sim.run(qcL, shots=shots, dynamic=True)
    print("M: sim circ on  backend=%s"%(backend_sim))
    result = job.result()
    for qc in qcL:
        counts = result.get_counts(qc)
        print('simu circ:%s, counts:'%qc.name,end=''); pprint(counts)
        print(qc.draw(output="text", idle_wires=False))
        verify(qc,counts)

    if not args.executeCircuit:
        print('NO execution of circuit, use -E to execute the job')
        exit(0)

    
    # - - - -  FIRE REAL JOB - - - - - - -
    print('M:IBMProvider()...')
    provider = IBMProvider()

    if 0:
        backend1 = least_busy(provider.backends(dynamic_circuits=True))
        print("M: least_busy backend:", backend1.name)

    # Make sure to use any OpenQASM 3-enabled backend.
    qasm3_backends = set( backend.name for backend in provider.backends(dynamic_circuits=True))
    print("M:The following backends support dynamic circuits:", qasm3_backends)
    assert args.backName in qasm3_backends
    backend = provider.get_backend(args.backName)

    # youtube tutorial uses: 'ibm_auckland'
    if 1:
        backend1 = least_busy(provider.backends(dynamic_circuits=True))
        print("M: least_busy backend:", backend1.name)
        #backend=backend1

    print('\nmy backend=',backend)
    print('base gates:', backend.configuration().basis_gates)

    
    qcLT = qk.transpile(qcL, backend, initial_layout=[args.qubit], seed_transpiler=args.rnd_seed,optimization_level=args.optimLevel)

    t0=time.time()
    job =  backend.run(qcLT,shots=shots, dynamic=True)
    jid=job.job_id()
    print('submitted JID=',jid,backend.name ,' now wait for execution of your circuit ...')
    job_monitor(job)
    t1=time.time()
    print('done elaT=%.1f (min)'%((t1-t0)/60.))
    result = job.result()
    for qc in qcLT:
        counts = result.get_counts(qc)
        print('transp circ:%s, counts:'%qc.name,end=''); pprint(counts)
        print(qc.draw(output="text", idle_wires=False))
        verify(qc,counts)

    #resD={ k:counts.get(k) for k in counts.keys()}
    # alternative: counts.items() --> dict_items([('000', 196), ('001', 799), ('010', 1110), ...
    #print('resD keys:',resD.keys())

    print('M:done')
