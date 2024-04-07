#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Demonstration of amplitude encoding
transpilation to naitive gates
execution on noisy fake simu
'''

from qiskit import QuantumCircuit , transpile #, execute
from qiskit_aer import AerSimulator
import numpy as np
from time import time

from qiskit.visualization import plot_state_qsphere
import matplotlib.pyplot as plt

# Number of qubits
nq = 7  # 11 qubits takes 28 sec to transpile
# Generate a random vector of size 2^nq
random_vector = np.random.rand(2**nq)

# Normalize the vector
data_vector = random_vector / np.linalg.norm(random_vector)

# Create a quantum circuit 
qc = QuantumCircuit(nq)

# Initialize the qubits to the classical data vector
qc.initialize(data_vector,  [i for i in range(nq)])
qc.measure_all()

backend = AerSimulator()
if 0:
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    #backName="ibm_torino"
    backName="ibm_hanoi"
    backend = service.get_backend(backName) 

# Transpile the circuit for the simulator
basis_gates = ['p','u3','cx']
T0=time()
qcT = transpile(qc, backend, basis_gates=basis_gates)
elaT=time()-T0
if nq<=4: print(qcT.draw(output='text',idle_wires=False))
cx_depth=qcT.depth(filter_function=lambda x: x.operation.num_qubits == 2 )
num_cx_gates = qcT.count_ops().get('cx')
print('nq=%d transp took %.1f sec, CX:  count=%d  depth=%d '%(nq,elaT,num_cx_gates,cx_depth))


# Execute the circuit
shots=50000
print('run %d qubit circ for %d shots on: %s ...'%(nq,shots,backend))
assert  shots >  data_vector.shape[0]*10
T0=time()
job =   backend.run(qcT, shots=shots)      
result = job.result() # Grab the results from the job.
elaT=time()-T0
counts = result.get_counts(0)
#print(counts)

# Convert counts to probabilities
probs = {k: v / shots for k, v in counts.items()}

print("elaT=%.1f sec, measured probabilities:"%elaT)
print('state  measProb  measAmpl  trueAmpl   diff')  
maxDiff=0
for i, tAmp in enumerate(data_vector):
    state = format(i, '0%db' % nq)  # Format i as a binary string of length nq
    if state in probs:
        mProb=probs[state]
    else:
        mProb=0.
    mAmp=np.sqrt(mProb);
    diff=mAmp - tAmp; adiff=abs(diff)
    if maxDiff < adiff: maxDiff = adiff
    print("%s:   %.3f      %.3f     %.3f   %7.3f" % (state, mProb , mAmp, tAmp, diff))
    if i>15: break

print('maxDiff=%.3f   1/sqrt(shots)=%.3f'%(maxDiff,1/np.sqrt(shots)))
