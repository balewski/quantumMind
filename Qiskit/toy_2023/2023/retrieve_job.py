#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
retrieve job by ID

https://medium.com/qiskit/qiskit-backends-what-they-are-and-how-to-work-with-them-fb66b3bd0463

'''
import time
from pprint import pprint
from qiskit import IBMQ
import qiskit
from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus

print('\nIBMQ account:'); pprint(IBMQ.load_account())
print('\nIBMQ providers:'); pprint(IBMQ.providers())

provider = IBMQ.get_provider(group='open')
backName='ibmq_santiago'

backend = provider.get_backend(backName)
print('\nmy backend=',backend)

jid='628ab6fc9c5eed8980c2a9c4'
jid='628aba162cf47d8047386959'

job=None
jobs = backend.jobs()
for xjob in jobs:
    xjid=xjob.job_id()
    print('see jid:',xjid)
    if xjid==jid :
        job=xjob; break
if job==None:
    print('job not found, exit')
    exit(13)

print('job IS found on %s, check status ...'%backName)
startT = time.time()
job_status = job.status()
while job_status not in JOB_FINAL_STATES:
    print('Status  %s , est. queue  position %d,  mytime =%d (sec)' %(job_status.name,job.queue_position(),time.time()-startT))
    #print(type(job_status)); pprint(job_status)
    time.sleep(10)
    job_status = job.status()

print('final job status, retriving it ...',job_status)
if job_status!=JobStatus.DONE :
    print('abnormal job termination', backend,', job_id=',xjid)
    exit(0)
jobHeadD = job.result().to_dict()
jobExpD=jobHeadD.pop('results')
print('jobRes header:'); pprint(jobHeadD)
assert jobHeadD['success']

numExp=len(jobExpD)
print('\nM: job found, num experiments=',numExp,', status=',jobHeadD['status'])
assert numExp>0

iexp=numExp//2
exp1=jobExpD[iexp]
print('explore experiment i=',iexp); pprint(exp1)
shots=exp1['shots']
runDate=jobHeadD['date']
elaTime=jobHeadD['time_taken']
print(runDate,' elapsed time:%.1f sec, shots=%d'%(elaTime,shots))

print('\n alternative unpacking of experiment')
result_exp = job.result()
counts_exp = result_exp.get_counts(0)
print('counts_exp:',type(counts_exp)); pprint(counts_exp)

print('result_exp',type(result_exp))
res_expD=result_exp.to_dict()
print('res_expD keys:',res_expD.keys())

print('\nM: get circuit as executed')
circEL=job.circuits() # circuits as-executed
circ4=circEL[iexp]
print('numCirc=',len(circEL),circ4.name)
print('exec circ depth:',circ4.depth(),', Gate counts:', circ4.count_ops())
print(circ4)

print("\n-----\n inspect some aspects of the submitted job:")
print('\nbackend options:'); pprint(job.backend_options())
print('\n job header'); pprint(job.header())

