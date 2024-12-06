#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

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

    parser.add_argument('-q','--sys_qubits',type=int,default=3, help="qubit count for 'system' U ")
    parser.add_argument('-t','--num_Tgate',type=int,default=2, help="number of Tgates in system U")
    parser.add_argument('-n','--num_shots',type=int,default=2000, help="shots")
    parser.add_argument('-N','--num_Pstrings',type=int,default=5, help="num pairs of Pauli strings")
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def make_systemU(nq, nT):
    """
    Create a random Clifford circuit,
    next add nT T-gates at the end  at random qubits
    
    Returns:
    QuantumCircuit: The quantum circuit implementing the unitary.
    """
    assert nq>0
    assert nT<=nq

    qc=random_clifford(nq).to_circuit()
    
    # Randomly choose distinct qubits without repetition
    chosen_qubits = random.sample(range(nq), nT)
    
    # Add a T-gate to each chosen qubit
    for qb in chosen_qubits:
        qc.t(qb)
    
    return qc

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
    if verb>0: print('Usys:'); print(Usys.definition.draw())
  
    P1str=random_pauli_string(nqs)
    P2str=random_pauli_string(nqs)
    #P1str='I'*nqs
    if verb>0: print('P1str:',P1str,'  P2str:',P2str)
    P1op = pauli_circuit(P1str).to_gate(label='P1')
    cP2op = controlled_pauli_circuit(P2str).to_gate(label='cP2')
    if verb>0: 
        print('P1:'); print(P1op.definition.draw())
        print('cP2:'); print(cP2op.definition.draw())

    # Initialize a new quantum circuit with n qubits
    qc3 = QuantumCircuit(2*nqs+1)
    #qc3.x(0);    return qc3,[P1str,P2str]
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
    
#=================================
#  M A I N
#=================================
if __name__ == "__main__":
    args=commandline_parser()
    #np.set_printoptions(precision=3)

    nqs=args.sys_qubits
    qcU=make_systemU(nqs,args.num_Tgate)
    # Convert the circuit to a gate
    Uop = qcU.to_gate(label='U')
    nq=2*nqs+1
    
    obs = SparsePauliOp("I" * (nq-1)+"Z")   # MSBF
    print('obs:',obs)

    backend = AerSimulator()
    print('job started,  nq=%d  at %s ...'%(nq,backend.name))
    options = EstimatorOptions()
    options.default_shots=args.num_shots
    estimator = Estimator(backend,options=options)

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

    evL=[0]*args.num_Pstrings
    T0=time()
    for j in range(args.num_Pstrings):
        qc,P1P2=make_otoc_circ(Uop,verb=j==0)
        if j==0: print(qc)
        qcT = pm.run(qc)
        obsT=obs.apply_layout(qcT.layout)
        job = estimator.run([(qcT,obsT)])    
        result = job.result()
        rdata=result[0].data
        evL[j]=abs(rdata.evs)
        if j==0:
            print("Metadata: ",result[0].metadata)
        if j<15 or j%100==0: print(j,"Expectation value: %7.3f +/- %.3f  P1P2: %s"%(rdata.evs,rdata.stds,P1P2))
    T1=time()
    avrEV=np.mean(evL)
    print('avrEV=%.2f'%(avrEV))
    measIN=-np.log(avrEV)
    trueIN=args.num_Tgate*np.log(4/3)
    print('Final I_N meas=%.2f   true=%.2f  numT=%d  N=%d  shots/circ=%d  elaT=%.1f min'%(measIN, trueIN,args.num_Tgate,args.num_Pstrings,args.num_shots,(T1-T0)/60.))
