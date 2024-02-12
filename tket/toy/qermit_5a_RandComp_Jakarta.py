#!/usr/bin/env python3
'''
Randomized compilation

Replace Folding(.)  with PauliFrameRandomisation(.)

'''

from pytket.extensions.quantinuum import QuantinuumBackend
from pytket import Circuit, OpType
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.tailoring import PauliFrameRandomisation
from pprint import pprint


#...!...!....................
def build_circuit_mcx(nq = 5):
    circuit = Circuit(n_qubits=nq)
    for i in range(nq-1): circuit.X(i)
    circuit.add_gate(OpType.CnX, [j for j in range(nq)])    
    return circuit


#...!...!....................
def add_measurements(c):    
    nq=c.n_qubits
    c.add_c_register(name='c', size=nq)
    qL=[ c.qubits[j] for j in range(nq)]
    c.add_barrier(qL)
    for j in range(nq):
        c.Measure(c.qubits[j],c.bits[j])

        
#...!...!....................
def print_qc(qc,text='myCirc'):
    cxDepth=qc.depth_by_type(OpType.CX)
    print('\n---- %s ---- CX depth=%d'%(text,cxDepth))
    print(tk_to_qiskit(qc))

#...!...!....................
def run_measurement(qc0):
    qc=qc0.copy()
    add_measurements(qc)    
    print('simu  shots=%d'%(shots),tk_backend) 
    handle = tk_backend.process_circuit(qc, n_shots=shots)
    counts = tk_backend.get_result(handle).get_counts()  
    #print('M:counts',counts)
    return counts
    


#=================================
#  M A I N 
#=================================

if __name__ == "__main__":
   
    shots=2000
    nRepeat=2
    optL=2
   
    qcM=build_circuit_mcx(4)  
    nq=qcM.n_qubits
    
    print_qc(qcM,'master circuit, %d shots'%shots)

    if 0:
        from pytket.extensions.qiskit import AerBackend
        tk_backend = AerBackend()
        machine='Aer'


    if 1:  # ------ run noisy IBMQ simulator
        from pytket.extensions.qiskit import IBMQEmulatorBackend
        tk_backend = IBMQEmulatorBackend( backend_name='ibmq_jakarta',instance='', )
        machine='FakeJakart'

    qcMT = tk_backend.get_compiled_circuit(circuit=qcM, optimisation_level=optL)
    print_qc(qcMT,'master transpiled')

    nCirc=1
  
    outL1=[]
    key0=(1,)*nq

    
    print('M: run original circuit %d times'%nCirc)
    qqDepth=qcMT.depth_by_type(OpType.CX)
    avr1=0
    for i in range(nCirc):
        print('run i=%d MT '%(i))
        countsL=run_measurement(qcMT)
        cnt=countsL[key0]
        avr1+=cnt
        outL1.append([qqDepth,cnt])
    avr1/=nCirc
    print('\nM: orig avr1=%.0d outL1=[cx-depth,counts]:'%avr1,outL1)
    if  machine=='Aer': exit(0)
    
    print('M: generate %d RC variants ...'%nCirc)
    qcRL=PauliFrameRandomisation().sample_circuits(circuit=qcMT, samples=nCirc)
    
    outL2=[]
    for i in range(nCirc):
        qcR=qcRL[i]
        print_qc(qcR,'RC full circ')
        qcRT = tk_backend.get_compiled_circuit(circuit=qcR, optimisation_level=optL)
        print_qc(qcRT,'RC transp circ')
        qqDepth=qcRT.depth_by_type(OpType.CX)
   
        print('run i=%d f  qqdDepth=%d'%(i,qqDepth))
        counts=run_measurement(qcRT)
        outL2.append([qqDepth,counts[key0]])

    print('\nM: orig circuit outL1=[cx-depth,counts]:',outL1)
    print('\nM: RC variants outL2=[cx-depth,counts]:',outL2)

