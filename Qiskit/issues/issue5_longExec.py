#!/usr/bin/env python3
''' problem: adding delay inside corcuit reults with 

'''
import qiskit as qk
from qiskit import IBMQ


backName='ibmq_jakarta'
#backName='ibmq_guadalupe'
print('\nIBMQ account - just pass load_account()'); IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='lbnl', project='chm170')
backend = provider.get_backend(backName)

print('\nmy backend=',backend)

# more info about qubits
backProps = backend.properties()
# Conversion factors from standard SI units
us = 1e6
ns = 1e9

# -------Create a Quantum Circuit 
nq=3
circ = qk.QuantumCircuit(nq,nq)
#circ.h(0)
#circ.barrier()
for q in range(nq):
    circ.barrier()
    circ.measure(q,q)
print(circ)
print('\n transpile\n')
circT = qk.transpile(circ, backend=backend, optimization_level=3, seed_transpiler=12, scheduling_method="alap")

print(circT)

clock_dt=backend.configuration().dt
print('duration of drive per qubit, clock dt/ns=%.2f'%(clock_dt*1e9))
tmax=0
for qid in range(nq):
    len_dt=circT.qubit_duration(qid)
    if tmax<len_dt: tmax=len_dt
    len_us=len_dt*clock_dt*1e6
    u2_len=backProps.gate_length('u2', qid)
    #print('q%d duration %.2f (us) =%d (dt)'%(qid,len_us,len_dt))
    print('q%d  T1/us=%.1f, T2/us=%.1f, U2_duration/ns %.2f '%(qid,t1*us, t2*us, u2_len*ns))
    #print('dur/us=%.2f'%(tmax*clock_dt*1e6))
    


