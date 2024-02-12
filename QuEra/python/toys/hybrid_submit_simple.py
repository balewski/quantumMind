#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


#https://github.com/amazon-braket/amazon-braket-examples/issues/347

# based on   https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs-first.html
from braket.aws import AwsQuantumJob
import os

deviceArn='arn:aws:braket:us-east-1::device/qpu/quera/Aquila'

# Set the environment variable
os.environ["AMZN_BRAKET_DEVICE_ARN"]=deviceArn

job = AwsQuantumJob.create(
    deviceArn,
    source_module="hybrid_task_simple.py",
    entry_point="hybrid_task_simple:run_hybrid_task",
    wait_until_complete=True
)
 
'''  
==========================  Good output =======================
...
 'rgrgrggrggr': 3,
 'rgrgrgrgggr': 3,
 'rgrgrgrgrer': 1,
 'rgrgrgrgrgr': 29,
 'rgrgrgrgrrr': 1,
 'rgrgrgrrgrg': 1}
cost tracker, charge/$: 0
{'arn:aws:braket:us-east-1::device/qpu/quera/Aquila': {'shots': 100,
                                                       'tasks': {'COMPLETED': 1}}}
M:Test job completed!!!!!
Code Run Finished
2023-09-13 04:26:54,972 sagemaker-training-toolkit INFO     Reporting training SUCCESS
'''
