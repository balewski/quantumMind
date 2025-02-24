#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit_aer import AerSimulator
from iqm.qiskit_iqm import IQMProvider

#...!...!....................
def create_ghz(n):    
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n, name="c")
    qc = QuantumCircuit(qr, cr)

    qc.h(0)
    for i in range(1, n):  qc.cx(0,i)
    
    qc.measure_all()
    return qc

nq=7

qc=create_ghz(nq)
print(qc)
print('M: ideal circ gates count:', qc.count_ops())

backend1 = AerSimulator()
print('job started,  nq=%d  at %s ...'%(qc.num_qubits,backend1.name))
options = SamplerOptions()
options.default_shots=10000


qcEL=(qc,)  # quant circ executable list
sampler = Sampler(mode=backend1, options=options)
job = sampler.run(qcEL)
result=job.result()
counts=result[0].data.meas.get_counts()
print('counts1:',counts)

print('access IQM backend ...')
# os.environ["IQM_TOKEN"] is set already
provider=IQMProvider(url="https://cocos.resonance.meetiqm.com/garnet")
backend2 = provider.get_backend()

print('got backend:',backend2.name)

qcT = transpile(qc, backend2)

print(qcT.draw('text', idle_wires=False))
print('M: transpiled GHZ(nq=%d) gates count:'%nq, qcT.count_ops())
qcETL=(qcT,) 
sampler = Sampler(mode=backend2, options=options)
job = sampler.run(qcETL)
result=job.result()
counts=result[0].data.meas.get_counts()
print('counts2:',counts)


