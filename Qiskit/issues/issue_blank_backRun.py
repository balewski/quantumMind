#!/usr/bin/env python3
''' problem: adding delay inside corcuit reults with 

'''

from qiskit import   QuantumCircuit, transpile
from qiskit.tools.monitor import job_monitor
from pprint import pprint
from qiskit_ibm_runtime import QiskitRuntimeService 

print('M: activate QiskitRuntimeService() ...')
service = QiskitRuntimeService()    
  

backName='ibm_osaka' # error: virtual_bit = final_layout_physical[i]
backName='ibm_cairo' #  works 
#backName='ibmq_qasm_simulator'
print('M: get backend:',backName)
backend = service.get_backend(backName)

print('\nmy backend=',backend)

#...!...!....................
def ghz_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):  qc.cx(0, i)
    qc.measure_all()
    return qc



# -------Create a Quantum Circuit 
qc=ghz_circuit(5)

print(qc.draw(output="text", idle_wires=False))
print('\n transpile for ',backend)
qcT = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=12) #, scheduling_method="alap")
print(qcT.draw(output='text',idle_wires=False))  # skip ancilla

if backName!="ibmq_qasm_simulator"  :
    # Get the layout mapping from logical qubits to physical qubits
    physQubitLayout = qcT._layout.final_index_layout(filter_ancillas=True)
    print('M:phys qubits ',physQubitLayout , backend)
job =  backend.run(qcT,shots=1000)
jid=job.job_id()

print('submitted JID=',jid,backend ,'\n now wait for execution of your circuit ...')
 
job_monitor(job)
counts = job.result().get_counts(0)
pprint(counts)

try:
    job =  provider.retrieve_job(jid)
except:
    print('job=%s  is NOT found, quit\n'%job)
    exit(99)
    
print('job IS found, retrieving it ...')

job_status = job.status()
print('Status  %s , queue  position: %s ' %(job_status.name,str(job.queue_position())))
print('M:ok')

