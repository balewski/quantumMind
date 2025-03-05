#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
from braket.aws.aws_quantum_task import AwsQuantumTask

from pprint import pprint
import json
task_arn='arn:aws:braket:us-east-1:765483381942:quantum-task/fa9c947e-b60a-49d6-9d3b-c923fa7353be'

try:
    job = AwsQuantumTask(arn=task_arn, poll_timeout_seconds=30)
except:
    print('job NOT found, quit\n ARN=',task_arn)
    exit(99)
    
print('job IS found',job)
state=job.state()
print('state:',state)

if job.state()!='COMPLETED':  exit(99)

results=job.result()
rawBitstr=results.get_counts()
pprint(rawBitstr)

outF='out.json'
with open(outF, 'w') as json_file:
    json.dump(rawBitstr, json_file)
print('shots saved '+outF)
