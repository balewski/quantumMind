#!/usr/bin/env python3
''' problem:  IBMProvider API is slow

IBMProvider(): 111, 47, 22, 16, 64, 62, 92  sec  (too long)
qk.transpile(circ, ..) : 97, 61, 7, 4, 47  sec   (too long)
circ exec+get results: 6, 35, 16 sec , OK

'''
import qiskit as qk
from qiskit.tools.monitor import job_monitor
from qiskit_ibm_provider import IBMProvider
from pprint import pprint
import time

backName='ibmq_qasm_simulator'
print('M:IBMProvider()...')
t1=time.time()
provider = IBMProvider()
t2=time.time()
print('got provider, took %.1f sec'%(t2-t1))

backend = provider.get_backend(backName)
t3=time.time()
print('got backend=%s  , took %.1f sec '%(backend,t3-t2))

# -------Create a Quantum Circuit 
circ = qk.QuantumCircuit(3,3)
circ.h(0)
circ.cx(0, 1) 
circ.cx(0, 2)
circ.barrier()
circ.measure(0,0)
print(circ)

t4=time.time()
circT = qk.transpile(circ, backend=backend, optimization_level=3, seed_transpiler=12)
t5=time.time()
print('transpiled, took %.1f sec'%(t5-t4))

print(circT)

job =  backend.run(circT,shots=1000)
jid=job.job_id()

print('submitted JID=',jid,backend ,'\n now wait for execution of your circuit ...')
 
job_monitor(job)
results=job.result()
counts = results.get_counts(0)
pprint(counts)
t6=time.time()
print('exec+counts, took %.1f sec'%(t6-t5))

# working w/ dict gives the most flexibility
headD = results.to_dict()
t7=time.time()
print('results-2-dict, took %.1f sec'%(t7-t6))
pprint(headD)

print('M:ok')

