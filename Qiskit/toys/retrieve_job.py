#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Retrieve and analyze completed quantum jobs with detailed execution metrics
"""


from pprint import pprint
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum")
jid='cwxwma160bqg008pxqbg'
#jid='ctss66g3zkm0008s9yqg' # has 'old' result format circuits but is DONE
job = service.job(jid)
print('status:',job.status(), type(job.status()),type(job))
backName=job.backend().name
jobMetr=job.metrics()
#1print('jobMetr:',jobMetr)
qa={}
qa['backend']=backName
qa['timestamp_running']=jobMetr['timestamps']['running']
qa['quantum_seconds']=jobMetr['usage']['quantum_seconds']
qa['all_circ_executions']=jobMetr['executions']
print('QA:',qa)

jobRes = job.result()
#1print('jobRes:',jobRes)
counts = jobRes[0].data.c0.get_counts()

print(counts)

nqc=len(jobRes)  # number of circuit in the job
print('M: got %d circ+results'%nqc, 'JID=',jid)


for ic in range(nqc):
    res=jobRes[ic]
    print(ic,'res:',res);
    for key, value in vars(res.data).items():
        print(f"{key}: {value}")
        num_shots=res.data[key].num_shots
        break
    print('nShot',num_shots)
    #?qc=job.circuits()[ic]  # transpiled circuit 
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

