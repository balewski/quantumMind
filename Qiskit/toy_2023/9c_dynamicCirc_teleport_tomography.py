#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
set 'secret' state using U3
use dynamic circuit for  alic-2-bob teleportation
do tomography on teleported secret state
No graphics

Generate series of circuit testing various intermedate results


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
from bitstring import BitArray

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="exec on real backend, may take long time ")
    parser.add_argument( "-d","--dumpQasm", action='store_true', default=False, help="saves transpiled circuit as QASM")
    parser.add_argument('-b','--backName',default='ibmq_jakarta',help="backand for computations, should support dynamic circuits" )
    parser.add_argument('-n','--numShots',type=int,default=5000, help="shots")
    parser.add_argument('-q','--qubits', default=[1,2,3], type=int,  nargs='+', help='1 or 2 qubits, space separated ')
    parser.add_argument('--secretState', default=[2.4,0.6], type=float,  nargs='+', help='initial state defined by U3("theta", "phi", "lam=0") ')
    parser.add_argument('-L','--optimLevel',type=int,default=1, help="transpiler ")
    #  * 0: no optimization * 1: light optimization * 2: heavy optimization * 3: even heavier optimization If None, level 1 will be chosen as default.
    args = parser.parse_args()

    args.rnd_seed=111 # for transpiler

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    #?assert len(args.qubits)==3
    assert len(args.secretState)==2
    return args



#...!...!....................
def circU3tomo(nq,axMeas):    
    theta=args.secretState[0]
    phi=args.secretState[1]
    qc= qk.QuantumCircuit(nq,nq,name='U3tomo')
    for iq in range(nq): # WARN: sign flip accumulates        
        if iq==1: theta=-theta
        if iq==2: phi=-phi
        # Creates secret quantum initial state
        qc.u(theta,phi,0.0,iq) # theta,phi,lambda
        meas_tomo(qc,iq,iq,axMeas)

    return qc


#...!...!....................
def meas_tomo(qc,tq,tb,axMeas):
    assert axMeas in 'xyz'
    if axMeas=='y':
        qc.sx(tq)

    if axMeas=='x':
        qc.rz(np.pi/2,tq)  # it should have been -pi/2
        qc.sx(tq)
        qc.rz(-np.pi/2,tq) # it should have been +pi/2
        
    qc.measure(tq,tb)
    qc.name+='_'+axMeas


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
        
#...!...!....................
def circTeleport():
    name='Teleport_q'+(''.join([str(x) for x in args.qubits])   )
    
    ## SETUP
    # Protocol uses 3 qubits and 2 classical bits in 2 different registers
    # include third register for measuring Bob's result
    qr = qk.QuantumRegister(3, name="q")
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
def verify_U3(countsD,tbit):  # LSBF
    print('\nMeasured expected values on tbit=%d for 3 tomo-axis:'%tbit)
    evD={}
    for axMeas in countsD:
        counts=countsD[axMeas]
        #print('axMeas',axMeas,counts)
        n0=0; n1=0
        for key in counts:
            #print('kkk', key,'msb:',key[-tbit-1],n1)
            if key[-tbit-1]=='1' :  # needs MSBF ???
                n1+=counts[key]
            else:
                n0+=counts[key]
        print(axMeas,tbit,'n0,n1',n0,n1)
        ns=n0+n1
        prob=n1/(n0+n1)
        n0=max(1,n0)
        n1=max(1,n1)        
        probEr=np.sqrt(n0*n1/ns)/ns
        ev=1-2*prob
        evEr=2*probEr
        evD[axMeas]=np.array([ev,evEr])
    #print('verify_U3 meas bit=',tbit,evD)

    #.... ground truth
    theta,phi=args.secretState
    
    if tbit==1:  theta=-theta
    if tbit==2:
        theta=-theta
        phi=-phi
        
    print('ver tomo true theta,phi',theta,phi)
    cth=np.cos(theta)
    sth=np.sin(theta)
    cphi=np.cos(phi)
    sphi=np.sin(phi)

    uz=cth
    ux=sth*cphi
    uy=sth*sphi

    evT={'z':uz,'y':uy,'x':ux}
    #print('True: ',evT)
    print('\nCompare expected values on tbit=%d for 3 tomo-axis:'%tbit)
    isOK=True
    nSigThr=4
    for ax in list('xyz'):
        evM,evE=evD[ax]
        dev=evM-evT[ax]
        nSig=abs(dev)/evE
        isOK*= nSig<nSigThr
        print('ax=%c  nSig=%.1f  true=%.3f, meas=%.3f +/- %.3f'%(ax,nSig,evT[ax],evM,evE) )    
    
    msg='    PASS    ' if isOK  else '   ***FAILED***'
    print('verify ',msg,' shots=%d per ax, nSigThr=%d\n'%(args.numShots,nSigThr))
    return isOK
    
#...!...!....................
def true_secret():    
    theta,phi=args.secretState
    cth=np.cos(theta/2.)
    sth=np.sin(theta/2.)
    cphi=np.cos(phi)
    sphi=np.sin(phi)
    

#...!...!....................
def analysis():    
    # ... sort results based on circ type ...
    countsDa={} # tomo-U3
    countsDb={} # tomo-tele
    for qc in qcL:
        counts = result.get_counts(qc)
        print('simu circ:%s, counts:'%qc.name,end=''); pprint(counts)
        print(qc.draw(output="text", idle_wires=False))
        axMeas=qc.name[-1]
        if 'U3tomo' in qc.name: countsDa[axMeas]=counts
        if 'Tele' in qc.name: countsDb[axMeas]=counts 
        if args.dumpQasm:
            print('M: dump QASM3 idela circ:\n')
            qiskit.qasm3.dump(qc,sys.stdout)
            print('\n  --- end ---\n')

    isOK=True     
    # verify tomo-U3, separately for each qubit
    for tbit in range(nq):
        isOK *= verify_U3(countsDa,tbit)

    # verify tomo-tele for the last qubit
    isOK *=verify_U3(countsDb,-1)  # final measurement is on MSB position
   
    msg='    PASS    ' if isOK  else '   ***FAILED***'
    print('M: verify ',msg)
    
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    np.set_printoptions(precision=3)
    
    nq=len(args.qubits)
    backend_sim = qk.Aer.get_backend('aer_simulator')
    shots=args.numShots

    # construct 3-q parallel circuits    
    qcL=[ circU3tomo(nq,axMeas)   for axMeas in list('zyx')]

    # teleportation circ w/ 3 tomo-projections
    for  axMeas in list('zyx'):
        qcTele=circTeleport() # Alic-Bob teleportation circuit
        meas_tomo(qcTele,2,2,axMeas)
        qcL.append(qcTele)
    #print(qcL[2],qcL[2].name); ok44
    
    job = backend_sim.run(qcL, shots=shots, dynamic=True)
    print("M: sim circ on  backend=%s"%(backend_sim))
    result = job.result()
    analysis()
     
    #?add_tomo_teleport
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

   
    if 1:
        backend1 = least_busy(provider.backends(dynamic_circuits=True))
        print("M: least_busy backend:", backend1.name)
        #backend=backend1
        #backend=backend_sim
        
    print('\nmy backend=',backend)
    print('base gates:', backend.configuration().basis_gates)

    
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
    analysis()

    print('M:done')
