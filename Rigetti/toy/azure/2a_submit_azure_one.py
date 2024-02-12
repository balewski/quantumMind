#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Submit 1 circuit to Rigetti  - CAREFULL w/ credits

'''
import time
import json
from pprint import pprint

from activate_azure import activate_azure_provider

import qiskit as qk

#...!...!..................
def create_circ( nq):  # Qiskit circ
    assert nq>0
    qc = qk.QuantumCircuit(nq, nq)
    qc.h(0)
    for j in range(1,nq):
        qc.cx(0,j)
    qc.barrier()
    qc.measure(range(nq), range(nq))
    return qc

#...!...!....................
def retrieve_qiskit_job(backend, job_id, verb=1):
    from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus
    print('\nretrieve_job backend=',backend,' search for',job_id)
    job=None
    try:
        job = backend.retrieve_job(job_id)
    except:
        print('job NOT found, quit\n')
        exit(99)

    print('job IS found, retrieving it ...')
    startT = time.time()
    job_status = job.status()
    while job_status not in JOB_FINAL_STATES:
        print('Status  %s , queue  position: %s,  %s ,  mytime =%d ' %(job_status.name,str(job.queue_position()),backend,time.time()-startT))
        #print(type(job_status)); pprint(job_status)
        time.sleep(15)
        job_status = job.status()

    print('final job status',job_status)
    if job_status!=JobStatus.DONE :
        print('abnormal job termination', backend,', job_id=',job_id)
        # ``status()`` will simply return``JobStatus.ERROR`` and you can call ``error_message()`` to get more
        print(job.error_message())
        exit(0)

    return  job


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    print('define some circ')
    qc=create_circ(3)
    print(qc)
  

    #machine='quantinuum.sim.h1-1e' # simulator
    machine='rigetti.sim.qvm'  # simulator
    #machine="rigetti.qpu.aspen-11"  #real HW
    backend = activate_azure_provider(machine)
    print('got machine=',machine)

    qcT= qk.transpile(qc, backend)
    print(qcT)
     
    job = backend.run(qcT, count=10)
    jobId=job.id()
    print("Job id:", jobId)
    #jobId='4a0aaa00-49e6-11ed-8d57-0242ac110003'

    job=retrieve_qiskit_job(backend, jobId)
        
    result = job.result()

    counts = result.get_counts()
    print('counts=',counts)

    print('M: done one job')

    headD = result.to_dict()
    pprint(headD)
