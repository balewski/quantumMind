#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 Retrieve  results of zoo-job  run on Quandela HW

Required INPUT:
    --expName: exp_j33ab44

Output:  raw  yields + meta data
'''

import time,os,sys
from pprint import pprint
import numpy as np
from toolbox.Util_H5io4 import  read4_data_hdf5, write4_data_hdf5

from time import time, sleep
import perceval as pcvl

from submit_zoo_job import harvest_results

from toolbox.Util_Quandela  import monitor_async_job
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase output verbosity", default=1)
    parser.add_argument("--basePath",default='out',help="head dir for set of experiments")
    parser.add_argument('-e',"--expName",  default='exp_62a15b79',help='IBMQ experiment name assigned during submission')

    args = parser.parse_args()

    args.inpPath=os.path.join(args.basePath,'jobs')
    args.outPath=os.path.join(args.basePath,'meas')
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
   
    return args

        
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    np.set_printoptions(precision=3)
    inpF=args.expName+'.ibm.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.inpPath,inpF),verb=args.verb)

    sbm=expMD['submit']
    pprint(sbm)

    if args.verb>1: pprint(expMD)

    nCirc=len(sbm['job_ids'])
    jid=sbm['job_ids'][-1]
                   
    # ------  construct sampler-job w/o backend ------
    proc = pcvl.RemoteProcessor(sbm['backend'])  # QPU name does not matter ??
    job = proc.resume_job(jid)
    print('Job status =%s name=%s'%(job.status(),job.name))
    monitor_async_job(job)
    print('M: last circ=%d finished'%nCirc)

    #.... pull all results
    nCirc=expMD[ 'submit']['num_circ']
    sampResL=[None]*nCirc
    for ic in range(nCirc):
        jid=sbm['job_ids'][ic]
        job = proc.resume_job(jid)
        assert job.status()=='SUCCESS'
        sampResL[ic] = job.get_results()

   
    harvest_results(sampResL,expMD,expD)

   
    if args.verb>2: pprint(expMD)
    
    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.meas.h5')
    write4_data_hdf5(expD,outF,expMD)


   # print('   ./postproc_zoo.py  --expName   %s   -p a b   -Y\n'%(expMD['short_name']))
  
    
    
