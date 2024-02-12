#!/usr/bin/env python3
''' problem: submit & retrieve job

'''
import qiskit as qk
from qiskit_ibm_provider import IBMProvider
from pprint import pprint
import time

backName='ibmq_qasm_simulator'
print('M:IBMProvider()...')
provider = IBMProvider()

backend = provider.get_backend(backName)
print('\nmy backend=',backend)


# -------Create a Quantum Circuit 
circ = qk.QuantumCircuit(3,3)
circ.h(0)
circ.cx(0, 1) 
circ.cx(0, 2)
circ.barrier()
circ.measure(0,0)
print(circ)

job =  backend.run(circ,shots=1000)
jid=job.job_id()

print('submitted JID=',jid,backend ,'\n now wait for execution of your circuit ...')
time.sleep(10)
job2 = provider.retrieve_job(jid)

print('job IS found, retrieving it ...')
