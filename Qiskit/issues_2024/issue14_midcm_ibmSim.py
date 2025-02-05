#!/usr/bin/env python3
''' problem:  only Aer & real HW understand mid-circ-measurement

'''
import qiskit as qk
from qiskit.tools.monitor import job_monitor
from qiskit_ibm_provider import IBMProvider
from pprint import pprint
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeJakarta

#...!...!....................
def access_fake_backend(backName, verb=1): 
    import importlib
    assert 'fake' in backName
    a,b=backName.split('_')
    b0=b[0].upper()
    backName='Fake'+b0+b[1:] #+'V2'
    print('zzz',a,b,backName)
    #module = importlib.import_module("qiskit.test.mock")
    module = importlib.import_module("qiskit.providers.fake_provider")
    cls = getattr(module, backName)
    return cls()


# -------Create a Quantum Circuit     
crPre = qk.ClassicalRegister(1, name="m_pre")
crPost = qk.ClassicalRegister(2, name="m_post")
qr = qk.QuantumRegister(2, name="q")
qc= qk.QuantumCircuit(qr, crPre,crPost)
qc.h(0)
qc.x(1)
qc.measure(0,crPre)
with qc.if_test((crPre, 0)):
    qc.x(0)
    qc.x(1)
qc.measure(0,crPost[0])
qc.measure(1,crPost[1])
# ....
print(qc)
print('\n transpile...\n')

if 1:
    backName='ibmq_qasm_simulator'
    #backName='ibmq_jakarta'
    print('M:IBMProvider()...')
    provider = IBMProvider()
    backend = provider.get_backend(backName)    

if 0:
    backend = AerSimulator.from_backend(FakeJakarta())

if 0:
    fakeBack=access_fake_backend('fake_jakarta')
    backend = AerSimulator.from_backend(fakeBack)
    
    
qubits=[2,3]
print('\nmy backend=',backend,'qubits=',qubits)



circT = qk.transpile(qc, backend=backend, initial_layout=qubits,optimization_level=1, seed_transpiler=12) #, scheduling_method="alap")
print(circT.draw(output="text", idle_wires=False))

print('M:ok')

