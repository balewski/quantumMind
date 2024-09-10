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
from qiskit.circuit import Parameter

#...!...!....................
def create_h_circuit(n):
    theta = Parameter('θ')
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n, name="c")
    qc = QuantumCircuit(qr, cr)

    qc.h(0)    
    for i in range(1, n):  qc.cx(0,i)
    qc.barrier()
    for i in range(0,n):  qc.measure(i,i)
    qc.reset(1)
    qc.ry(theta,1)
    qc.measure(1,0)
    return qc,theta

nq=4
qcP,thetaP=create_h_circuit(nq)
print(qcP)
print(qcP.draw('text', idle_wires=False))

options = SamplerOptions()
options.default_shots=10000

backName='ibm_kyoto'
print('\n repeat on  %s backend ...'%backName)
service = QiskitRuntimeService(channel="ibm_quantum")

backend2 = service.backend(backName)
print('use backend =', backend2.name )
pm = generate_preset_pass_manager(optimization_level=3, backend=backend2)
qcT = pm.run(qcP)
print('transpiled for',backend2)
print(qcT.draw('text', idle_wires=False))

qcE=qcT.assign_parameters({thetaP:0.33})
print(qcE.draw('text', idle_wires=False))

sampler = Sampler(backend=backend2, options=options)
qcEL=(qcE,) 
job = sampler.run(qcEL)
print('job submitted to ', backend2.name)
result=job.result()

counts=result[0].data.c.get_counts()
print('counts:',counts)



