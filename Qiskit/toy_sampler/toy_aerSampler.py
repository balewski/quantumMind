#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
#from qiskit_ibm_runtime import EstimatorV2 as Estimator

#...!...!....................
def create_h_circuit(n):
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n, name="c")
    qc = QuantumCircuit(qr, cr)

    qc.h(0)
    for i in range(1, n):  qc.cx(0,i)
    qc.barrier()
    for i in range(0,n):  qc.measure(i,i)
    return qc

nq=4
qc=create_h_circuit(nq)
print(qc)

backend1 = AerSimulator()
print('job started,  nq=%d  at %s ...'%(qc.num_qubits,backend1.name))
options = SamplerOptions()
options.default_shots=10000


qcEL=(qc,)  # quant circ executable list
sampler = Sampler(backend=backend1, options=options)
job = sampler.run(qcEL)
result=job.result()

counts=result[0].data.c.get_counts()
print('counts:',counts)


print('\n repeat on  fake backend ...')
service = QiskitRuntimeService(name="olcf")
backName='ibm_torino'
#backName='ibm_kyoto'
noisy_backend = service.get_backend(backName)
backend2 = AerSimulator.from_backend(noisy_backend)

print('use noisy_backend =', noisy_backend.name )
pm = generate_preset_pass_manager(optimization_level=1, backend=backend2)
qcT = pm.run(qc)
print('transpiled for',backend2)
print(qcT.draw('text', idle_wires=False))

sampler = Sampler(backend=backend2, options=options)
qcEL=(qcT,) 
job = sampler.run(qcEL)
result=job.result()

counts=result[0].data.c.get_counts()
print('counts:',counts)

