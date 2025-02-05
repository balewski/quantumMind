#!/usr/bin/env python3
''' problem:  retireve job given ID

'''

from qiskit_ibm_provider import IBMProvider

backName='ibmq_qasm_simulator'
print('M:IBMProvider()...')
provider = IBMProvider()
job_id='cg4c4hcs7e0p5tv35o0g'
#job_id="4d8cccfc-e364-4e86-9e2c-ad3407d16c4c"
#job_id="a72db405-6966-439c-ac62-414e07384de7"
#job_id="cfvbv4mtm3omfc7i2lfg"; backName='ibmq_jakarta'
backend = provider.get_backend(backName)

print('\nretrieve_job backend=',backend,' search for',job_id)
job=None
try:
    job = backend.retrieve_job(job_id)
except:
    print('job=%s  is NOT found, quit\n'%job)
    exit(99)
    
print('job IS found, retrieving it ...')

job_status = job.status()
print('Status  %s , queue  position: %s ' %(job_status.name,str(job.queue_position())))


