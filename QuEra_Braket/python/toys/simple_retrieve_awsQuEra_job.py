#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
bare-bone AWS-QuEra job retrival

Use case:
 ./simple_retrieve_awsQuEra_job.py -t arn:aws:braket:us-east-1:765483381942:quantum-task/d85f4cc2-af61-43b4-b013-fd6fd146d009


'''

import time,os,sys
from pprint import pprint
from braket.aws.aws_quantum_task import AwsQuantumTask


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase output verbosity", default=1)
    parser.add_argument('-t',"--taskARN",  default='arn:aws:braket:us-east-1:765483381942:quantum-task/b0feb2c4-d1fd-45d7-8839-f3553895a2cf',help='AWS-QuEra experiment ARN  assigned during submission')       

    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def retrieve_aws_job( task_arn, verb=1):  # AWS-QuEra job retrieval
    #print('\nretrieve_job search for ARN',task_arn)
    job=None
    try:
        job = AwsQuantumTask(arn=task_arn, poll_timeout_seconds=30)
    except:
        print('job NOT found, quit\n ARN=',task_arn)
        exit(99)
    
    print('job IS found, retrieving it ..., state=',job.state())
    if job.state() in ['CANCELLING', 'CANCELLED', 'FAILED']:
        print('retrieval ABORTED for ARN:%s\n'%task_arn)
        exit(0)
    
    startT = time.time()    
    while job.state()!='COMPLETED':
        print('State  %s,    mytime =%d, wait ... ' %(job.state(),time.time()-startT))
        time.sleep(30)
        
    print('final job status:',job.state())
        
    return  job 
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    
    #arn='arn:aws:braket:us-east-1:765483381942:quantum-task/874bf405-8720-4e77-9fec-b6b84bfa5016' #   'status': 'COMPLETED'
    arn='arn:aws:braket:us-east-1:765483381942:quantum-task/1830c574-3744-44f8-8bee-9af4e70a0757'  # state CANCELLED
    
    arn=args.taskARN
    job=retrieve_aws_job( arn)

    print('\nM:task counts:'); pprint(job.result().get_counts())
    
    if args.verb>1: print('\nM:task meta-data:'); pprint(job.metadata())    
    print('DONE')


