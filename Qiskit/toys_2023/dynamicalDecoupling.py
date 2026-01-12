#!/usr/bin/env python3
''' 
enable dynamical decopupling
the patern of X-X insertion for initial state=0 qubits depends on the schedul: ALAP vs. ASAP

'''
import qiskit as qk
from qiskit.tools.monitor import job_monitor
from qiskit import IBMQ
import pprint as pprint

#...!...!....................
def make_ghz_circ(nq):
    name='ghz_%dq'%nq
    ghz = qk.QuantumCircuit(nq, nq,name=name)
    ghz.h(0)
    ghz.h(nq-1)
    for idx in range(1,nq):
        ghz.cx(0,idx)
    
    ghz.barrier(range(nq))
    ghz.measure(range(nq), range(nq))
    print(ghz)
    return ghz

backName='ibmq_lima'
print('\nIBMQ account - just pass load_account()'); IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='lbnl', project='chm170')
backend = provider.get_backend(backName)
print('\nmy backend=',backend)


# -------Create a Quantum Circuit
nq=3
circ=make_ghz_circ(nq)
sched='alap'
sched='asap'
print('\n transpile, sched=%s\n'%sched)
circT = qk.transpile(circ, backend=backend, optimization_level=3, seed_transpiler=12, scheduling_method=sched)
print(circT)

#....
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
durations = InstructionDurations.from_backend(backend)
#1print('durations'); print(durations)

# balanced X-X sequence on all qubits
dd_sequence = [XGate(), XGate()]

pm = PassManager([ALAPSchedule(durations),
                  DynamicalDecoupling(durations, dd_sequence)])
circ_dd = pm.run(circT)
print('dd circ:',type(circ_dd))
print(circ_dd)
