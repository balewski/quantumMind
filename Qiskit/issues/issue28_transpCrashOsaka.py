#!/usr/bin/env python3
'''  transpiler crashes for Osaka

'''

from qiskit import   QuantumCircuit, transpile
from pprint import pprint
from qiskit_ibm_runtime import QiskitRuntimeService 

print('M: activate QiskitRuntimeService() ...')
service = QiskitRuntimeService()    
  

backName='ibm_osaka' # ??? error: virtual_bit = final_layout_physical[i]
#backName='ibm_cairo' #  works 
#backName='ibm_torino' # works 
#backName='ibm_hanoi'  #???
#backName='ibmq_qasm_simulator' # works
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

initial_layout=[41, 59, 53, 60, 43, 42, 40]

# -------Create a Quantum Circuit 
qc=ghz_circuit(7)

print(qc.draw(output="text", idle_wires=False))
print('\n transpile for ',backend)
qcT = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=12,initial_layout=initial_layout) #, scheduling_method="alap")
print(qcT.draw(output='text',idle_wires=False))  # skip ancilla

if backName!="ibmq_qasm_simulator"  :
    # Get the layout mapping from logical qubits to physical qubits
    physQubitLayout = qcT._layout.final_index_layout(filter_ancillas=True)
    print('M:phys qubits ',physQubitLayout , backend)
job =  backend.run(qcT,shots=1000)
jid=job.job_id()

print('submitted JID=',jid,backend ,'\n now wait for execution of your circuit ...')
 
print('M:ok')

