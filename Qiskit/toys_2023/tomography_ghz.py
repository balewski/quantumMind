#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Reconstruct GHZ state tomography
2 qubit are used 

No graphics

Updated 2023-02

Based on https://qiskit.org/documentation/experiments/tutorials/state_tomography.html

'''

import time,os,sys
from pprint import pprint

import numpy as np
import qiskit as qk
from packaging import version

from qiskit.tools.monitor import job_monitor
from qiskit_ibm_provider import least_busy
from qiskit_ibm_provider import IBMProvider

from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import StateTomography

# For simulation
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeParis


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="exec on real backend, may take long time ")
    parser.add_argument('--secretState', default=[2.4,0.6], type=float,  nargs='+', help='initial state defined by U3("theta", "phi", "lam=0") ')
    parser.add_argument('-b','--backName',default='ibmq_jakarta',help="backand for computations" )
    parser.add_argument('-n','--numShots',type=int,default=20100, help="shots")
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
def circU3(nq):
    qc= qk.QuantumCircuit(nq,nq)
    iq=0
    theta=args.secretState[0]
    phi=args.secretState[1]
    # Creates secret quantum initial state
    qc.u(theta,phi,0.0,iq) # theta,phi,lambda

    return qc

#...!...!....................
def circ_init():
   # GHZ State preparation circuit
    nq = 2
    qc_ghz = qk.QuantumCircuit(nq)
    qc_ghz.h(0)
    qc_ghz.s(0)
    for i in range(1, nq):
        qc_ghz.cx(0, i)
    return qc_ghz
    
    
#...!...!....................
def ana_exp(qcL,resultL):
    evD={}
    for qc in qcL:
        counts = resultL.get_counts(qc)
        print('circ:%s, counts:'%qc.name,end=''); pprint(counts)
        n0=counts['0']; n1=counts['1']
        ns=n0+n1
        prob=n1/(n0+n1)
        n0=max(1,n0)
        n1=max(1,n1)        
        probEr=np.sqrt(n0*n1/ns)/ns
        ev=1-2*prob
        evEr=2*probEr
        axMeas=qc.name[0]
        evD[axMeas]=(ev,evEr)
        #print(axMeas,'n0,n1',n0,n1)
    return evD
    
#...!...!....................
def verify(evD,txt):    
    theta,phi=args.initState
    cth=np.cos(theta)
    sth=np.sin(theta)
    cphi=np.cos(phi)
    sphi=np.sin(phi)

    uz=cth
    ux=sth*cphi
    uy=sth*sphi

    evT={'z':uz,'y':uy,'x':ux}
    #1print('True: ',evT)

    print('\nCompare expected values on %s for 3 tomo-axis:'%txt)
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
    
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    backend_sim = qk.Aer.get_backend('aer_simulator')
    shots=args.numShots

    qcInp=circU3(1)
    # QST Experiment, Quantum State Tomography
    qstexp1 = StateTomography(qcInp)

    # definition: https://qiskit.org/documentation/experiments/stubs/qiskit_experiments.library.tomography.StateTomography.html
    qcL=qstexp1.circuits()
    
    print('nqc',len(qcL))
    for qc in qcL[:3]:   print('ideal',qc.name); print(qc)

    qcLT = qk.transpile(qcL, backend_sim, basis_gates=['p','sx','cx'] , seed_transpiler=args.rnd_seed,optimization_level=args.optimLevel)
    
    for qc in qcLT[:3]:
        print('transp',qc.name); print(qc)
    ok0
    
    qcL=[]
    for axMeas in list('zyx'):
        qc=circ_tomo(axMeas)
        #print(qc.name); print(qc)
        qcL.append(qc)

    job = backend_sim.run(qcL, shots=shots)
    print("M: sim circ on  backend=%s"%(backend_sim))
    result = job.result()

    
    for qc in qcL:
        counts = result.get_counts(qc)
        print('simu circ:%s, counts:'%qc.name,end=''); pprint(counts)
        print(qc.draw(output="text", idle_wires=False))

    evD=ana_exp(qcL,result)
    #1print('EV',evD)
    verify(evD,backend_sim)
        
    if not args.executeCircuit:
        print('NO execution of circuit, use -E to execute the job\n')
        exit(0)

    
    # - - - -  FIRE REAL JOB - - - - - - -
    print('M:IBMProvider()...')
    provider = IBMProvider()

    qasm_backends = set( backend.name for backend in provider.backends())
    print("M:The following backends are accessible:", qasm_backends)
    assert args.backName in qasm_backends
    backend = provider.get_backend(args.backName)

    if 1:
        backend1 = least_busy(provider.backends())
        print("M: least_busy backend:", backend1.name)
        #backend=backend1
        #backend=backend_sim

    print('\nmy backend=',backend.name)
    print('base gates:', backend.configuration().basis_gates)

    
    qcLT = qk.transpile(qcL, backend, initial_layout=[args.qubit], seed_transpiler=args.rnd_seed,optimization_level=args.optimLevel)

    t0=time.time()
    job =  backend.run(qcLT,shots=shots)
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

    evD=ana_exp(qcLT,result)
    #1print('EV',evD)
    verify(evD,backend.name)
    
    print('M:done')
