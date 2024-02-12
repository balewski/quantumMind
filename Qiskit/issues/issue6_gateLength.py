#!/usr/bin/env python3
''' problem: print duration of measurement: expected 5.35 (us) =24080 (dt)
'''
import qiskit as qk
from qiskit import IBMQ
from pprint import pprint

backName='ibmq_jakarta'
print('\nIBMQ account - just pass load_account()'); IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='lbnl', project='chm170')
backend = provider.get_backend(backName)

print('\nmy backend=',backend)

# more info about qubits
backProps = backend.properties()
# Conversion factors from standard SI units
us = 1e6
ns=1e9

defaults = backend.defaults()
config = backend.configuration()
inst_sched_map = defaults.instruction_schedule_map
measure = inst_sched_map.get('measure', qubits=config.meas_map[0])
print('meas_dur/dt=',measure.duration)


nq=backend.configuration().n_qubits
for qid in range(nq):
    t1=backProps.t1(qid)
    t2=backProps.t2(qid)

    sx_len=backProps.gate_length('sx', qid)
    print('q%d  T1/us=%.1f, T2/us=%.1f, sx_duration/ns %.2f '%(qid,t1*us, t2*us, sx_len*ns))
    
    


