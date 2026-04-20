#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

Check for running jobs at 
   https://us-east-1.console.aws.amazon.com/braket
'''

from braket.aws import AwsQuantumJob
from braket.jobs.config import OutputDataConfig,InstanceConfig,S3DataSourceConfig

import os
import  hashlib

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    job_name='jan-'+hashlib.md5(os.urandom(32)).hexdigest()[:6]
    print('S:submitting hybrid job:',job_name)

    deviceArn='arn:aws:braket:us-east-1::device/qpu/quera/Aquila'
    source_module='task_code/'  # local dir with all code within it

    hyperparams = {  # may contain only strings
        "num_iter": "3",
        "quera_device_arn": deviceArn,
        #"backend_short": "emul",  # skip HW
        "backend_short": "aquila",  # use HW
        "short_name": job_name,
        "num_shots": 45
    }

    # FIX:  add full S3 path to hpar , to be printed by the job
    
    # pick 1 of 3 entry points
    #entry_point="task_code.main1_task" #  hyperparam avaliable via env SM_TRAINING_ENV, Running Code As Subprocess, PREFERED - only checks passing arguments

    #OR:
    entry_point="task_code.main2_ahs" #  hyperparam avaliable via env SM_TRAINING_ENV, Running on Aquila on emulator 
    
    # OR:
    #entry_point="task_code.main2_task:execute_task", # Running Code As Process, hyperparam --> args, WORKS? NO
    
    
    # Create an AWS instance configuration
    instance_config = InstanceConfig(  instanceType="ml.m5.large", instanceCount=1, volumeSizeInGb=20  )

    # define S3 input & output
    BUCKET_NAME='amazon-braket-balewski-13977183483'
    s3base=f"s3://{BUCKET_NAME}/"
    input_data=S3DataSourceConfig(s3base+'dir-2/')
    output_data_config=OutputDataConfig(s3Path=s3base+'dir-5c/')

    job = AwsQuantumJob.create(
        deviceArn,
        job_name=job_name,
        source_module=source_module,        
        entry_point=entry_point,
        hyperparameters=hyperparams,
        instance_config=instance_config,
        output_data_config=output_data_config,
        input_data=input_data,
        wait_until_complete=True
    )
