#!/usr/bin/env python3
'''

'''
from pprint import pprint
from time import time
from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import  FakeCusco

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ionq import IonQProvider

#...!...!....................
def my_circuit(n):
    theta = Parameter('θ')
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n, name="meas")
    qc = QuantumCircuit(qr, cr)
    qc.h(0)
    qc.ry(theta,1)
    for i in range(1, n):  qc.cx(0,i)
    qc.barrier()
    for i in range(0,n):  qc.measure(i,i)
    #qc.reset(1)
    #qc.measure(1,0)
    return qc,theta


#...!...!....................
def run_job(qcP,backend,th1):
    print('use backend =', backend.name )
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    qcT = pm.run(qcP)
    qcE=qcT.assign_parameters({thetaP:th1})
    print('transpiled for',backend)
    print(qcE.draw('text', idle_wires=False))

    print('job started,  nq=%d  at %s ...'%(qcT.num_qubits,backend.name))

    qcEL=(qcE,)  # quant circ executable list
    sampler = Sampler(mode=backend)
    T0=time()
    job = sampler.run(qcEL)
    result=job.result()
    elaT=time()-T0
    counts=result[0].data.meas.get_counts()
    print('counts:',counts)
    print('run time %.1f sec'%(elaT))

# = = = = = = = = = = = =
#  M A I N 
# = = = = = = = = = = = =

nq=4
qcP,thetaP=my_circuit(nq)
print(qcP.draw('text', idle_wires=False))

backend = AerSimulator()
run_job(qcP,backend,0.33)

backend = FakeCusco() 
run_job(qcP,backend,0.88)


provider = IonQProvider()
backend_ionq = provider.get_backend("simulator")
#backend_ionq.set_options(noise_model="forte-1")
backend_ionq.set_options(noise_model="aria-1")
run_job(qcP,backend_ionq,0.44)



