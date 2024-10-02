#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit.circuit import Parameter

#...!...!....................
def add_randomH(qc,cr,qanc,qtrg):
    qc.reset(qanc)
    qc.h(qanc)
    qc.measure(qanc, 0)
    with qc.if_test((cr,1)):  # good for scaling
        qc.h(qtrg)

#...!...!....................
def create_h_circuit(n):
    theta = Parameter('θ')
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n, name="c")
    qc = QuantumCircuit(qr, cr)

    qc.h(0)
    for i in range(1, n):  qc.cx(0,i)
    qc.barrier()
    add_randomH(qc,cr,qanc=1,qtrg=2)
    qc.barrier()
    for i in range(0,n):  qc.measure(i,i)
    qc.reset(1)
    qc.ry(theta,1)
    qc.measure(1,0)
    return qc,theta

nq=4
qcP,thetaP=create_h_circuit(nq)
#print(qcP)
print(qcP.draw('text', idle_wires=False))

backend1 = AerSimulator()
print('job started,  nq=%d  at %s ...'%(qcP.num_qubits,backend1.name))
options = SamplerOptions()
options.default_shots=10000

qcE=qcP.assign_parameters({thetaP:0.33})
qcEL=(qcE,)  # quant circ executable list
sampler = Sampler(mode=backend1, options=options)
job = sampler.run(qcEL)
result=job.result()

counts=result[0].data.c.get_counts()
print('counts:',counts)


print('\n repeat on  fake backend ...')
service = QiskitRuntimeService(channel="ibm_quantum")

backName='ibm_torino'  # EPLG=0.7%
backName='ibm_kyiv'    # EPLG=1.4%
backName='ibm_nazca'   # EPLG=3.2%

noisy_backend = service.backend(backName)
backend2 = AerSimulator.from_backend(noisy_backend)

print('use noisy_backend =', noisy_backend.name )
pm = generate_preset_pass_manager(optimization_level=1, backend=backend2)
qcT = pm.run(qcP)
print('transpiled for',backend2)
print(qcT.draw('text', idle_wires=False))

sampler = Sampler(mode=backend2, options=options)
qcE2=qcT.assign_parameters({thetaP:0.33})
qcEL=(qcE2,) 
job = sampler.run(qcEL)
result=job.result()

counts=result[0].data.c.get_counts()
print('counts:',counts)

