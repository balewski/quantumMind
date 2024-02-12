#!/usr/bin/env python3
''' problem: adding delay inside corcuit reults with 

'''
import qiskit as qk
from qiskit.tools.monitor import job_monitor
from qiskit_ibm_provider import IBMProvider
from pprint import pprint

#backName='ibmq_jakarta'
backName='ibmq_qasm_simulator'
print('M:IBMProvider()...')
provider = IBMProvider()

backend = provider.get_backend(backName)
print('\nmy backend=',backend)


# -------Create a Quantum Circuit 
circ = qk.QuantumCircuit(3,3)
circ.h(0)
circ.cx(0, 1) 
#circ.delay(400)
circ.cx(0, 2)
circ.barrier()
circ.measure(0,0)
print(circ.draw(output="text", idle_wires=False))
print('\n transpile\n')
circT = qk.transpile(circ, backend=backend, optimization_level=3, seed_transpiler=12) #, scheduling_method="alap")
print(circT)
if 0:
    for iq in range(6):
        print('iq=%d duratin/dt=%d'%(iq,circT.qubit_duration(iq)))

job =  backend.run(circT,shots=1000)
jid=job.job_id()

print('submitted JID=',jid,backend ,'\n now wait for execution of your circuit ...')
 
job_monitor(job)
counts = job.result().get_counts(0)
pprint(counts)

try:
    job = backend.retrieve_job(jid)
except:
    print('job=%s  is NOT found, quit\n'%job)
    exit(99)
    
print('job IS found, retrieving it ...')

job_status = job.status()
print('Status  %s , queue  position: %s ' %(job_status.name,str(job.queue_position())))
print('M:ok')

