#!/usr/bin/env python3
# WORKS if job was run with qiskit-ibm-provider, reported as 'circuit-runner'
# NOT working for jobs run with  qiskit-ibm-runtime, aka 'sampler'

from pprint import pprint
from qiskit_ibm_provider import IBMProvider
import qiskit.qasm3  # only to dump OpenQasm3 circuits
import sys

provider = IBMProvider()
jid='cfnt39flq5sjub2afvhg'  # small fid-forward circ
jid='cpx9kgpv9m800087ktf0'  # 4q GHZ
jid='/cpxcrvvn8rag008qk2y' # q5 GHz
#jid='cpsxxed0f6rg008x6080'  # 27 qubit job
job = provider.retrieve_job(jid)

# Retrieve the results from the job
jobRes = job.result()
resL=jobRes.results  
nqc=len(resL)  # number of circuit in the job
counts=jobRes.get_counts()
if nqc==1: counts=[counts]  # this is poor design
print('M: got %d circ+results'%nqc, 'JID=',jid)


for ic in range(nqc):
    qc=job.circuits()[ic]  # transpiled circuit 
    resHead=resL[ic].header # auxiliary info about used hardware
    print('\nM: circ=%d %s'%(ic,qc.name))
    print('result header: ',end=''); pprint(resHead)
    print('counts:',counts[ic])
    cx_depth=qc.depth(filter_function=lambda x: x.operation.name == 'cx')
    q2_depth=qc.depth(filter_function=lambda x: x.operation.num_qubits > 1)
    print('circuit q2_depth=%d ,gate count:'%q2_depth,dict(qc.count_ops()))
 
    print(qc.draw(output="text", idle_wires=False,fold=140))
    if 1:
        print('M: dump QASM3 idela circ:\n')
        qiskit.qasm3.dump(qc,sys.stdout)
        print('\n  --- end ---\n')

    if ic>1: break
print('M:OK')

