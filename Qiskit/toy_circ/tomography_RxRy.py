#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Reconstruct R(theta) mography

No graphics

Updated 2023-02

Based on https://qiskit.org/documentation/experiments/tutorials/state_tomography.html

'''

import time,os,sys
from pprint import pprint

import numpy as np
import qiskit as qk
from packaging import version


#from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import StateTomography

# For simulation
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeParis


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-Rx', default=None, type=float, help='Rx(theta) or None')
    parser.add_argument('-Ry', default=None, type=float, help='Rx(theta) or None')
    
    parser.add_argument('-n','--numShots',type=int,default=20100, help="shots")

    parser.add_argument('-L','--optimLevel',type=int,default=1, help="transpiler ")
    #  * 0: no optimization * 1: light optimization * 2: heavy optimization * 3: even heavier optimization If None, level 1 will be chosen as default.
    args = parser.parse_args()

    args.rnd_seed=111 # for transpiler

    print('qiskit ver=',qk.__qiskit_version__)

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
def circRx():
    nq=1
    qc= qk.QuantumCircuit(nq)
    
    iq=0
    theta=args.Rx
    # Creates secret quantum initial state
    qc.rx(theta,iq) # theta,phi,lambda
    #phi=-np.pi/2
    #qc.u(theta,phi,0.0,iq) # theta,phi,lambda

    return qc

#...!...!....................
def circRy():
    nq=1
    qc= qk.QuantumCircuit(nq)
    
    iq=0
    theta=args.Ry
    # Creates secret quantum initial state
    qc.ry(theta,iq) # theta,phi,lambda

    return qc

    
#...!...!....................
def ana_exp(qcL,resultL):
    evD={}
    axMap={'(0,)':'z','(1,)':'x','(2,)':'y'}
    for qc in qcL:
        txt=qc.name.split('_')[1]
        #print('tt',txt)

        axMeas=axMap[txt]
        counts = resultL.get_counts(qc)
        print('circ:%s, axMeas=%s counts:'%(qc.name,axMeas),end=''); pprint(counts)
        n0=1;n1=1  # set minimal counts for error setimation
        if '0' in counts: n0=counts['0']
        if '1' in counts: n1=counts['1']
        ns=n0+n1
        prob=n1/(n0+n1)
        probEr=np.sqrt(n0*n1/ns)/ns
        ev=1-2*prob
        evEr=2*probEr
        
        evD[axMeas]=np.array([ev,evEr])
        #print(axMeas,'n0,n1',n0,n1)
    return evD
    
#...!...!....................
def verify(evD,txt):

    if args.Rx!=None:
        theta=args.Rx; phi=-np.pi/2
    if args.Ry!=None:
        theta=args.Ry; phi=0.
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
    np.set_printoptions(precision=3)

    backend_sim = qk.Aer.get_backend('aer_simulator')
    shots=args.numShots

    if args.Rx!=None:  qcInp=circRx()
    if args.Ry!=None:  qcInp=circRy()

    assert args.Rx!=None or args.Ry!=None
    print(qcInp)
    # QST Experiment, Quantum State Tomography
    qstexp1 = StateTomography(qcInp)

    # definition: https://qiskit.org/documentation/experiments/stubs/qiskit_experiments.library.tomography.StateTomography.html
    qcL=qstexp1.circuits()
    
    print('M: num tomo circuits:',len(qcL))
    for qc in qcL:   print('ideal',qc.name); print(qc)

    qcLT = qk.transpile(qcL, backend_sim, basis_gates=['p','sx','cx'] , seed_transpiler=args.rnd_seed,optimization_level=args.optimLevel)
    
    #for qc in qcLT:  print('M:transp',qc.name); print(qc)
        
    job = backend_sim.run(qcLT, shots=shots)
    print("M: sim circ on  backend=%s"%(backend_sim))
    result = job.result()

 
    for qc in qcLT:
        counts = result.get_counts(qc)        
        print('simu circ:%s, counts:'%qc.name,end=''); pprint(counts)
        print(qc.draw(output="text", idle_wires=False))

    evD=ana_exp(qcLT,result)
    print('M:expectation values:',evD)
    verify(evD,backend_sim)
        
    
    
    print('M:done')
