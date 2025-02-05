#!/usr/bin/env python3
''' 
Execution of  grover.amplify(problem) takes forever
Answer:
the Sampler primitive from terra is not efficient since it use the statevector formalism to perform simulations and it makes the statevector evolves gate by gate which can be costly. I highly recommend you to use the primitives from Aer, it can use by default more sophisticated simulation methods: 
 
from qiskit_aer.primitives import Sampler as AerSampler
grover = Grover(sampler=AerSampler(), iterations=numIter)
 
For 21 qubits the simulation took approximately 0.15 min, jan on laptop 0.19 min

'''
from qiskit import QuantumCircuit
from qiskit.algorithms import AmplificationProblem, Grover
#from qiskit.primitives import Sampler  # inefficient
from qiskit_aer.primitives import Sampler as AerSampler
import time

#...!...!....................
def load_matching_oracle(qc,qs,qt,qanc,nq_data, phase=True):
    #... XNOR over data qubits (aka OM)
    for j in range(nq_data ):
        qc.cx(qs+j,qt+j)
        qc.x(qt+j)
    # OA
    if phase:
        qc.x(qanc) ;   qc.h(qanc)
    qL=[ qt+j  for j in range(nq_data )]    
    qc.mcx( qL, qanc )
    if phase:
        qc.x(qanc) ;   qc.h(qanc)
        
    # un-compute  (aka OM dag)
    for j in range(nq_data ):        
        qc.x(qt+j)
        qc.cx(qs+j,qt+j)
        
#...!...!....................
def callable_good_state(bitstr):
    if bitstr[-1] == "1":
        return True
    return False

#...!...!....................
def scan_results(resultL, thres):
    keyM={} # matched    
    probM=0.

    for mbits in resultL:
        prob=resultL[mbits]
        if prob<thres: continue
        keyM[mbits]=prob
        probM+=prob           

    print('%d winners probM=%.3f keyM:'%(len(keyM),probM),sorted(keyM))
  
#=================================
#  M A I N 
#=================================
        
nq=15  # works
nq=21  # 4000 slower?
numIter=4

inpF='circ_stat_prep_%dq.qasm'%nq
circ_state_prep=QuantumCircuit.from_qasm_file(inpF)
print(circ_state_prep)


circ_oracle=QuantumCircuit(nq)
if nq==15:
    qt=7; nq_data=4
if nq==21:
    qt=10; nq_data=6
load_matching_oracle(circ_oracle,0,qt,nq-1,nq_data)
print(circ_oracle)

problem = AmplificationProblem(
    circ_oracle, circ_state_prep,
    is_good_state=callable_good_state )


grover = Grover(sampler=AerSampler(), iterations=numIter)
print('Starting grover.amplify(problem) %d qubits ....'%nq)
T0=time.time()
result = grover.amplify(problem)
T1=time.time()
print('M: grover.amplify  elaT=%.2f min\n'%((T1-T0)/60.))
   
print('Result type:', type(result))
top_meas = result.top_measurement
numSol=len(result.circuit_results[0])
maxProb=result.max_probability
print('all solutions:', numSol)
print('Top measurement:',top_meas, 'prob=%.3f  numIter=%d'%(maxProb, numIter))

scan_results(result.circuit_results[0], maxProb/2)

