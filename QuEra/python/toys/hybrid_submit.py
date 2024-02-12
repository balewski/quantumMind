#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


#https://github.com/amazon-braket/amazon-braket-examples/issues/347

# based on   https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs-first.html
from braket.aws import AwsQuantumJob
import os
import  hashlib

from braket.jobs.config import OutputDataConfig,InstanceConfig,S3DataSourceConfig
 
BUCKET_NAME='amazon-braket-balewski-13977183483'
job_name='jan-'+hashlib.md5(os.urandom(32)).hexdigest()[:6]

deviceArn='arn:aws:braket:us-east-1::device/qpu/quera/Aquila'
hyperparams = {
    "num_iter": "5",
    "deviceArn": deviceArn,
    #"backend_name": "emul",
    "backend_name": "aquila",
    "short_name": job_name
}

# Create an instance configuration
instance_config = InstanceConfig(
    instanceType="ml.m5.large", instanceCount=1, volumeSizeInGb=20
)


# define S3 input & output
s3base=f"s3://{BUCKET_NAME}/"
#input_data=S3DataSourceConfig(f"s3://{BUCKET_NAME}/jobs/{job_name}/inp")
input_data=S3DataSourceConfig(s3base+'dir-2/')
#input_data=S3DataSourceConfig(s3base)
# {'dataSource': {'s3DataSource': {'s3Uri': 's3://....'}}}

#output_data_config=OutputDataConfig(s3Path=f"s3://{BUCKET_NAME}/jobs/{job_name}/out")
output_data_config=OutputDataConfig(s3Path=s3base+'dir-5/')
#output_data_config=OutputDataConfig(s3Path=s3base)

#code_location=s3base+'/dir-4'
# If source_module is an S3 URI, it must point to a tar.gz file. Otherwise, source_module may be a file or directory.


''' cost
https://aws.amazon.com/sagemaker/pricing/

Standard Instances	vCPU	Memory	Price per Hour
ml.t3.medium	2	4 GiB	$0.05
ml.m5.large	2	8 GiB	$0.115

https://amazon-braket-sdk-python.readthedocs.io/en/stable/_apidoc/braket.aws.aws_quantum_job.html

'''

job = AwsQuantumJob.create(
    deviceArn,
    job_name=job_name,
    hyperparameters=hyperparams,
    output_data_config=output_data_config,
    input_data=input_data,
    #code_location=code_location,  # Unable to download code,  HeadObject operation: Not Found
    source_module="hybrid_task.py",
    entry_point="hybrid_task:run_hybrid_task",
    instance_config=instance_config,
    wait_until_complete=True
)

# CALL: /usr/local/bin/python3.10 braket_container.py --backend_name aquila --deviceArn arn:aws:braket:us-east-1::device/
