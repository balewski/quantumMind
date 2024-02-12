#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Exercise all modes of computation for GHZ circuit, including connection to IBM

for IBM token go to: https://quantum-computing.ibm.com/account

based on 1_getting_started_with_qiskit.ipynb
Updated 2022-05
'''

import numpy as np
from pprint import pprint
import qiskit as qk
from qiskit import IBMQ
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor

print('qiskit ver=',qk.__qiskit_version__)

# -------Create a Quantum Circuit acting on a quantum register of three qubits
circ = qk.QuantumCircuit(3)
circ.h(0) # Add a H gate on qubit 0, putting this qubit in superposition.
circ.cx(0, 1) # now  qubits 0,1  in a Bell state.
circ.cx(0, 2) # now we have 3-qubit Bell state
print(circ)


print('\n ------ Statevector backend')

# Run the quantum circuit on a statevector simulator backend
backend = qk.Aer.get_backend('statevector_simulator')
# Create a Quantum Program for execution 
job = qk.execute(circ, backend)
result = job.result()
outputstate = result.get_statevector(circ, decimals=3)
print(outputstate)

print( '\n------- Unitary backend')
# Run the quantum circuit on a unitary simulator backend
backend = qk.Aer.get_backend('unitary_simulator')
job = qk.execute(circ, backend)
result = job.result()
# Show the results
print(result.get_unitary(circ, decimals=3))

print('\n -------- Add the  measurement, reuse circuit')
meas = qk.QuantumCircuit(3, 3)
meas.barrier(range(3))
# map the quantum measurement to the classical bits
meas.measure(range(3),range(3))
qc = circ+meas # The Qiskit circuit object supports composition
print(qc)

print("\n ------- Use Aer's qasm_simulator for counting")
backend_sim = qk.Aer.get_backend('qasm_simulator')
job_sim = qk.execute(qc, backend_sim, shots=1024)

# Grab the results from the job.
result_sim = job_sim.result()
counts = result_sim.get_counts(qc)
print('counts=',counts)
pprint(counts)


print('\n ----------Running circuits from the IBM Q account: ')

if 0: # for IBM token go to: https://quantum-computing.ibm.com/account
    #myToken='296..dc6'
   
    IBMQ.delete_account() # Delete the saved account from disk.
    # enable_account(TOKEN, HUB, GROUP, PROJECT): Enable your account in the current
    IBMQ.enable_account(myToken, hub='ibm-q-ornl', group='lbnl', project='chm170')
    IBMQ.save_account(myToken,overwrite=True)
    print('\nstored account:'); pprint(IBMQ.stored_account())
    print('\nactive account:'); pprint(IBMQ.active_account())
    
print('\nIBMQ account - just pass load_account()'); IBMQ.load_account()
print('\nIBMQ providers:'); pprint(IBMQ.providers())


if 0 :  # open
    provider = IBMQ.get_provider(group='open')
    backName='ibmq_santiago' # 5Q,  line-topo
    #backName='ibmq_lima'  # 5Q, T-topo
else:   # berkely
    provider = IBMQ.get_provider(hub='ibm-q-ornl', group='lbnl', project='chm170')
    #backName='ibmqx_hpc_qasm_simulator'
    backName='ibmq_toronto'  # 27 qubits


print('\n  provider:'); pprint(provider.backends())
backend = provider.get_backend(backName)
print('\nmy backend=',backend)
print(backend.status().to_dict())
print('propertioes keys:',backend.properties().to_dict().keys())
hw_qubits=backend.properties().to_dict() #'qubits')#
print('num hw qubits=',len(hw_qubits),hw_qubits.keys())
hw_config=backend.configuration().to_dict()
print('configuration keys:',hw_config.keys(),'n_qubits=',hw_config['n_qubits'],backend.configuration().n_qubits)

print('execute the circuit:'); print(qc)

job_exp = qk.execute(qc, backend=backend, shots=1024)
jid=job_exp.job_id()
print('submitted JID=',jid,backend )
#exit(0)  # quit here if you do not want to wait for results

job_monitor(job_exp)

result_exp = job_exp.result()
counts_exp = result_exp.get_counts(qc)
print(backName,'counts_exp:',type(counts_exp))
pprint(counts_exp)

