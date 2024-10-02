#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
use dynamic circuit for : alic-bob teleportation experiment
No graphics
My notes: https://docs.google.com/document/d/1gkGamdCdXWOyiEBcTgcF1h3ovXB-1vfETAyAbwlzJ68/edit?usp=sharing

Updated 2023-02
based on :
https://quantum-computing.ibm.com/services/programs/docs/runtime/manage/systems/dynamic-circuits/Teleportation

'''

import time,os,sys
from pprint import pprint

import numpy as np
import qiskit as qk
from packaging import version

from qiskit.tools.monitor import job_monitor
from qiskit_ibm_provider import least_busy
from qiskit_ibm_provider import IBMProvider
import qiskit.qasm3


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="exec on real backend, may take long time ")
    parser.add_argument( "-d","--dumpQasm", action='store_true', default=False, help="saves transpiled circuit as QASM")
    parser.add_argument('-b','--backName',default='ibmq_kolkata',help="backand for computations, should support dynamic circuits" )
    parser.add_argument('-n','--numShots',type=int,default=2002, help="shots")
    parser.add_argument('-q','--qubits', default=[1,2,3], type=int,  nargs='+', help='1 or 2 qubits, space separated ')
    parser.add_argument('--secretState', default=[2.4,0.], type=float,  nargs='+', help='initial state defined by U3("theta", "phi", "lam=0") ')
    parser.add_argument('-L','--optimLevel',type=int,default=1, help="transpiler ")
    #  * 0: no optimization * 1: light optimization * 2: heavy optimization * 3: even heavier optimization If None, level 1 will be chosen as default.
    parser.add_argument('--axMeas',type=str,default='z',choices=['x','y','z'], help="Bob's measurement axis")
    args = parser.parse_args()

    args.rnd_seed=111 # for transpiler

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert len(args.qubits)==3
    assert len(args.secretState)==2
    return args

#...!...!....................
def alice_send(qc, qs, qa, crz, crx):
    """Measures qubits a & b and 'sends' the results to Bob"""
    qc.cx(qs, qa)
    qc.h(qs)
    qc.measure(qs,crz)
    qc.measure(qa,crx)
    qc.barrier()

#...!...!....................
# This function takes a QuantumCircuit (qc), integer (qubit)
# and ClassicalRegisters (crz & crx) to decide which gates to apply
def bob_gates(qc, qubit, crz, crx):
    # Here we use qc.if_test to control our gates with a classical
    # bit instead of a qubit
    with qc.if_test((crx, 1)):
        qc.x(qubit)
    with qc.if_test((crz, 1)):
        qc.z(qubit)
    qc.barrier()

#...!...!....................
def meas_tomo(qc,axMeas):
    qb=2  # tmp
    crb=2
    assert axMeas in 'xyz'
    if axMeas=='y':
        qc.sx(qb)
        
    if axMeas=='x':
        qc.rz(-np.pi/2,qb)
        qc.sx(qb)
        qc.rz(np.pi/2,qb)
        
    qc.measure(qb,crb)

    
#...!...!....................
def circTeleport(nq):
    name='Teleport_q'+(''.join([str(x) for x in args.qubits])   )

    
    ## SETUP
    # Protocol uses 3 qubits and 2 classical bits in 2 different registers
    # include third register for measuring Bob's result
    qr = qk.QuantumRegister(nq, name="q")
    crz, crx = qk.ClassicalRegister(1, name="mz_alice"), qk.ClassicalRegister(1, name="mx_alice") # Alice
    crb = qk.ClassicalRegister(1, name="bob")  # Bob
    qc = qk.QuantumCircuit(qr, crz, crx, crb,name=name)

    qs=0;qa=1;qb=2  # soft ids of qubits: secrte, a+b form Bell state

    # Creates secret quantum initial state
    qc.u(args.secretState[0],args.secretState[1],0.0,qs)
    qc.barrier()
    
    # Creates a bell pair in qc using qubits a & b
    qc.h(qa) # Put qubit a into state |+>
    qc.cx(qa,qb) # CNOT with a as control and b as target
    qc.barrier()

    # Alice measures secret state in Z & X basis using one of bell-qubits
    alice_send(qc, qs, qa, crz, crx)

    # Bob receives 2 classical bits and applies gates on bell-state b-qubit
    bob_gates(qc, qb, crz, crx)

    return qc
    

#...!...!....................
def true_secret():    
    theta,phi=args.secretState
    cth=np.cos(theta/2.)
    sth=np.sin(theta/2.)
    cphi=np.cos(phi)
    sphi=np.sin(phi)
    
#...!...!....................
def verify_teleport(qc,counts,axMeas):
    pprint(counts)
    cnt={'0':0,'1':0}
    for key in counts:
        cnt[key[0]]+= counts[key]

        
    n0,n1=max(1,cnt['0']), max(1,cnt['1'])
    shots=n0+n1
    pop=n1/shots
    print('tele sum',n0,n1)
    popEr=np.sqrt(n0*n1/shots)/shots

    tpop=1.-np.cos(args.secretState[0]/2.)**2
    c1= np.abs(tpop-pop) < 3*popEr
    print('verify tele ax=%s pop=%.2f +/- %.2f  truth=%.2f'%(axMeas,pop,popEr,tpop))
    msg='    PASS    ' if c1   else '   ***FAILED***'
    print('verify  c1:',c1,msg,qc.name,'\n')
    
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    nq=len(args.qubits)
    backend_sim = qk.Aer.get_backend('aer_simulator')
    shots=args.numShots
    
    qcTele=circTeleport(nq) # Alic-Bob teleportation circuit
    
    meas_tomo(qcTele,args.axMeas)
         
    qcL=[qcTele]
    job = backend_sim.run(qcL, shots=shots, dynamic=True)
    print("M: sim circ on  backend=%s"%(backend_sim))
    result = job.result()
    for qc in qcL:
        counts = result.get_counts(qc)
        print('simu circ:%s, counts:'%qc.name,end=''); pprint(counts)
        print(qc.draw(output="text", idle_wires=False))
        if 'Tele' in qc.name:
            verify_teleport(qc,counts,args.axMeas)            
            if args.dumpQasm:
                print('M: dump QASM3 idela circ:\n')
                qiskit.qasm3.dump(qc,sys.stdout)
                print('\n  --- end ---\n')
    #M: dump QASM3
    
    if not args.executeCircuit:
        print('NO execution of circuit, use -E to execute the job\n')
        exit(0)

    
    # - - - -  FIRE REAL JOB - - - - - - -
    print('M:IBMProvider()...')
    provider = IBMProvider()

    # Make sure to use any OpenQASM 3-enabled backend.
    qasm3_backends = set( backend.name for backend in provider.backends(dynamic_circuits=True))
    print("M:The following backends support dynamic circuits:", qasm3_backends)
    assert args.backName in qasm3_backends
    backend = provider.get_backend(args.backName)

       
    print('\nmy backend=',backend)
    print('M:base gates:', backend.configuration().basis_gates)

    
    qcLT = qk.transpile(qcL, backend, initial_layout=args.qubits, seed_transpiler=args.rnd_seed,optimization_level=args.optimLevel)

    if args.dumpQasm:
        qc=qcLT[2]
        print(qc.draw(output="text", idle_wires=False))
        print('M: dump QASM3 transpiled circ:\n')
        qiskit.qasm3.dump(qc,sys.stdout)
        print('\n  --- end ---\n')
    
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
        print('transp circ:%s, nq=%d '%(qc.name,qc.num_qubits),end='')
        print(qc.draw(output="text", idle_wires=False))
        if 'Tele' in qc.name: verify_teleport(qc,counts,args.axMeas)

    print('M:done')
