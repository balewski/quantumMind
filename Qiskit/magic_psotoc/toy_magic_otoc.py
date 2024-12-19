#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
Measure 'magic' for selected unitaries
Method: OTOC = out-of-time-ordered correlator for Pauli strings
https://arxiv.org/pdf/2408.01663

*** random unitaries ***
1 block of 3 qubits U with 3 T-gates
./toy_magic_otoc.py --Utype rnd 3 1 --U_qubits 3   
>>> Magic: m=0.82+/-0.04  t=0.86 ; Tcnt: m=2.8+/-0.1  t=3 NPstr=800

2 blocks of 3 qubits U with 2 T-gates, total 4 gates
./toy_magic_otoc.py --Utype rnd 2 2 --U_qubits 3   

>>> Magic: m=0.52+/-0.03  t=1.15 ; Tcnt: m=1.8+/-0.1  t=4 

*** QFT  ***
./toy_magic_otoc.py --Utype qft --U_qubits 5 -N2000

*** angle encoding ***
./toy_magic_otoc.py --Utype ang-enc --U_qubits 9

 thV=np.linspace(-1,1,n)*np.pi
 for j in range(n): qc.ry(thV[j],j)

*** prob encoding ***
./toy_magic_otoc.py --Utype prob-enc --U_qubits 9

 thV=np.arccos(np.linspace(-1,1,n))
 for j in range(n): qc.ry(thV[j],j)

'''

from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime.options.estimator_options import EstimatorOptions
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
import random
import numpy as np
from time import time

from qiskit.quantum_info import random_clifford

import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int, help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-q','--U_qubits',type=int,default=3, help="qubit count for U ")
    parser.add_argument('-n','--num_shots',type=int,default=3000, help="shots")
    parser.add_argument('-N','--num_Pstrings',type=int,default=1000, help="num pairs of Pauli strings")
    parser.add_argument('-U','--Utype',nargs='+',default=['rnd', 3 ], help="type of Unitary with optional params, blankl separated ")
    args = parser.parse_args()
   
    if args.Utype[0]=='rnd':  # par: nTgate [nBlock]
        npar=len(args.Utype)
        assert npar>1
        # num Tgates per block
        args.rndU_nT=int(args.Utype[1])
        # number of U-blocks 
        if npar==3: args.rndU_nB=int(args.Utype[2])
        else: args.rndU_nB=1
    args.Utype=args.Utype[0]
    assert args.Utype in ['rnd','qft','ang-enc','ev-enc','mcx']
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def prepU_MCX( n):
    assert n>=2
    from qiskit.circuit.library import MCXGate
    qc = QuantumCircuit(n)
    #qc.mcx(range(1,n), 0)
    tgL=[i for i in range(1,n)]
    qc.mcx(tgL, 0)
    # theory : https://arxiv.org/pdf/1212.5069 
    return qc,4*(n-2)

#...!...!....................
def prepU_angEnc( n): 
    ''' angle encoding
    theta= x*pi  for x in [-1,1]
    '''
    qc = QuantumCircuit(n)
    
    thV=np.linspace(-1,1,n)*np.pi
    for j in range(n): qc.ry(thV[j],j)
    return qc,1
    
#...!...!....................
def prepU_evEnc( n): 
    ''' prob encoding
    theta= arcos(x)  for x in [-1,1]
    '''
    qc = QuantumCircuit(n)
    
    thV=np.arccos(np.linspace(-1,1,n))
    for j in range(n): qc.ry(thV[j],j)
    return qc,1
    
#...!...!....................
def prepU_qft( n): 
    qc = QuantumCircuit(n)
    ''' skip input
    # For example, prepare a sample state |0011> (binary for 3)
    qc.x(0)
    qc.x(1)
    
    Apply the QFT on the first n qubits of circuit qc.
    '''
    for j in range(n):
        # Apply H-gate on the j-th qubit
        qc.h(j)
        
        # Apply controlled-phase gates
        for k in range(j+1, n):
            # The phase is pi/2^(k-j)
            qc.cp(np.pi / 2**(k-j), k, j)
    
    # Finally, reverse the qubit order
    for i in range(n//2):
        qc.swap(i, n - i - 1)

    tTcnt=n*np.log(n)
    return qc,tTcnt

#...!...!....................
def prepU_random(nq, nT,nB=1):
    """
    Create a random Clifford circuit,
    next add nT T-gates at the end  at random qubits
    optional: repeat it nB times
    
    Returns:
    QuantumCircuit: The quantum circuit implementing the unitary.
    """
    assert nq>0
    assert nT<=nq
    assert nB>=1

    qc= QuantumCircuit(nq)
    for b in range(nB):
        
        if b>0 and 0: # OFF insert GHZ
            qc.h(0)
            for i in range(1,nq): qc.cx(0,i)
        qcr=random_clifford(nq).to_circuit().to_gate()
        qcr.name='clif_%d'%b
        qc.append(qcr, qargs=range(nq))
        # Randomly choose distinct qubits without repetition
        chosen_qubits = random.sample(range(nq), nT)       
        # Add a T-gate to each chosen qubit
        for qb in chosen_qubits: qc.t(qb)
    #1print(qc.decompose())
    return qc,nT*nB

#...!...!....................
def random_pauli_string(n):
    """Generate a random Pauli string of length n."""
    paulis = ['I', 'X', 'Y', 'Z']
    pauli_string = [random.choice(paulis) for _ in range(n)]
    return pauli_string

#...!...!....................
def controlled_pauli_circuit(pauli_string):
    """
    Create a QuantumCircuit that:
    - Initializes qubit 0 in the |+> state.
    - Applies controlled-Pauli gates from qubit 0 to qubits 1 to n.
    """
    n = len(pauli_string)
    total_qubits = n + 1  # Including the control qubit
    qc = QuantumCircuit(total_qubits)

    for i, p in enumerate(pauli_string):
        target_qubit = i + 1  # Since qubit 0 is the control qubit
        if p == 'I':
            pass  # Identity operator; no action needed
        elif p == 'X':
            qc.cx(0, target_qubit)
        elif p == 'Y':
            qc.cy(0, target_qubit)
        elif p == 'Z':
            qc.cz(0, target_qubit)
        else:
            raise ValueError(f"Invalid Pauli operator: {p}")
    return qc

#...!...!....................
def pauli_circuit(pauli_string):
    """
    Create a QuantumCircuit that:
    - Takes a Pauli string of length n.
    - Applies the corresponding Pauli gates (X, Y, Z) directly to qubits [0..n-1].
    - Ignores 'I' (identity), leaving that qubit unchanged.
    """
    n = len(pauli_string)
    qc = QuantumCircuit(n)  # Only n qubits, no control qubit

    for i, p in enumerate(pauli_string):
        if p == 'I':
            pass  # No operation needed
        elif p == 'X':
            qc.x(i)
        elif p == 'Y':
            qc.y(i)
        elif p == 'Z':
            qc.z(i)
        else:
            raise ValueError(f"Invalid Pauli operator: {p}")

    return qc

#...!...!....................
def make_otoc_circ(Usys,verb=1):
    ''' out-of-time-ordered correlator, based on https://arxiv.org/pdf/2408.01663
    sysU size: nqs
    nq= 2*nqs+1
    q0 is measured
    '''
    nqs=Usys.num_qubits
    Uinv=Usys.inverse(); Uinv.name='Uinv'
    if verb>0 and nqs<6: print('Usys:'); print(Usys.definition.draw())
  
    P1str=random_pauli_string(nqs)
    P2str=random_pauli_string(nqs)
    #P1str='I'*nqs
    if verb>0: print('P1str:',P1str,'  P2str:',P2str)
    P1op = pauli_circuit(P1str).to_gate(label='P1')
    cP2op = controlled_pauli_circuit(P2str).to_gate(label='cP2')
    if verb>1: 
        print('P1:'); print(P1op.definition.draw())
        print('cP2:'); print(cP2op.definition.draw())

    # Initialize a new quantum circuit with n qubits
    qc3 = QuantumCircuit(2*nqs+1)
  
    #initial state
    qc3.h(0)
    for i in range(nqs): qc3.h(i+1+nqs)
    for i in range(nqs): qc3.cx(i+1+nqs,i+1)
    qc3.barrier()
    
    qc3.append(cP2op, qargs=range(nqs+1))
    qsL=[i for i in range(1,nqs+1)]
    qc3.append(Usys, qargs=qsL)
    qc3.append(P1op, qargs=qsL)
    qc3.append(Uinv, qargs=qsL)
    qc3.barrier()
    qc3.x(0)
    qc3.append(cP2op, qargs=range(nqs+1))
    qc3.x(0)
    qc3.h(0)
    #print(qc3.decompose())
    #print(qc3)
    return qc3,[P1str,P2str]

#...!...!....................
def export_CSV_rndU():
    print('\nU qubits,num T-gate /rndCliff,num rndCliff,num T-gate,true magic,meas magic,err,meas num T-gate,err,num Pauli str,Utype,elaT/min')
    print('\n%d,%d,%d,%d,%.2f,%.2f,%.2f,%.1f,%.1f,%d,%s,%.1f'% \
          (nqU,args.rndU_nT,args.rndU_nB,args.rndU_nT*args.rndU_nB,tMagic,mMagic,mMagicErr,mTcnt,mTcntErr,args.num_Pstrings,args.Utype,elaT))

#=================================
#  M A I N
#=================================
if __name__ == "__main__":
    args=commandline_parser()
    np.set_printoptions(precision=3)

    nqU=args.U_qubits
    if args.Utype=='rnd':
        qcU,trueTcnt=prepU_random(nqU,args.rndU_nT,args.rndU_nB)  # random U + some Tgates
    if args.Utype=='qft': qcU,trueTcnt=prepU_qft(nqU)
    if args.Utype=='ang-enc': qcU,trueTcnt=prepU_angEnc(nqU)
    if args.Utype=='ev-enc': qcU,trueTcnt=prepU_probEnc(nqU)
    if args.Utype=='mcx': qcU,trueTcnt=prepU_MCX(nqU)

    
    # Convert the circuit to a gate
    Uop = qcU.to_gate(label='U')
    nq=2*nqU+1
    
    obs = SparsePauliOp("I" * (nq-1)+"Z")   # MSBF
    print('obs:',obs)

    backend = AerSimulator()
    print('job started,  nq=%d  at %s ...'%(nq,backend.name))
    options = EstimatorOptions()
    options.default_shots=args.num_shots
    estimator = Estimator(backend,options=options)

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

    evA=np.zeros(args.num_Pstrings)
    T0=time(); lastEV=0
    for j in range(args.num_Pstrings):
        qc,P1P2=make_otoc_circ(Uop,verb=j==0)
        if j==0 and  nqU<6 : print(qc)
        qcT = pm.run(qc)
        obsT=obs.apply_layout(qcT.layout)
        job = estimator.run([(qcT,obsT)])    
        result = job.result()
        rdata=result[0].data
        evA[j]=abs(rdata.evs)
        if j==0:
            print("Metadata: ",result[0].metadata)
        if j<15 : print(j,"Expectation value: %7.3f +/- %.3f  P1P2: %s"%(rdata.evs,rdata.stds,P1P2))
        if j%200==100 :
            avrEV=np.mean(evA[:j]); stdEV=np.std(evA[:j])/np.sqrt(j)
            delAvr=avrEV-lastEV; nSig=abs(delAvr)/stdEV
            lastEV=avrEV
            print('%4d Avr EV: %7.3f+/-%.3f  delEV=%6.3f  nSig=%.1f  elaT=%.1f min'%(j,avrEV,stdEV,delAvr,nSig,(time()-T0)/60.))
        
    T1=time()
    avrEV=np.mean(evA); stdEV=np.std(evA)/np.sqrt(evA.shape[0])
    #print('avrEV=%.2f'%(avrEV))
    mMagic=-np.log(avrEV)
    mMagicErr=stdEV/avrEV  # for y=log(x) sig(y)= sig(x)/x
    tMagic=trueTcnt*np.log(4/3)
    
    mTcnt=mMagic/np.log(4/3)
    mTcntErr=mMagicErr/np.log(4/3)
    elaT=(T1-T0)/60.
    print('Magic: m=%.2f+/-%.2f  t=%.2f ; Tgate cnt: m=%.1f+/-%.1f  t=%.1f ; Utype=%s nqU=%d  NPstr=%d  shots/circ=%d  elaT=%.1f min'%(mMagic,mMagicErr, tMagic,mTcnt,mTcntErr,trueTcnt,args.Utype,nqU,args.num_Pstrings,args.num_shots,elaT))

    if args.Utype=='rnd': export_CSV_rndU()
