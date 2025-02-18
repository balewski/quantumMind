#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 Retrieve  results of Quandela job

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

#from toolbox.Util_QiskitV2 import pack_counts_to_numpy
from submit_quandela_job import harvest_sampler_results

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
    
    inpF=args.expName+'.ibm.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.inpPath,inpF),verb=args.verb)

    sbm=expMD['submit']
    pprint(sbm)

    if args.verb>1: pprint(expMD)

    if 0:    #example decode one Qasm circuit
        rec2=expD['circQasm'][1].decode("utf-8") 
        print('qasm circ:',type(rec2),rec2)
    
    jid=sbm['job_ids'][-1]
    #1jid='cns2cfhqygeg00879yv0' # smapler,  cairo

    # ------  construct sampler-job w/o backend ------
    proc = pcvl.RemoteProcessor(sbm['backend'])  # QPU name does not matter ??
    job = proc.resume_job(jid)
    print('Job status =%s name=%s'%(job.status(),job.name))
    T0=time()
    i=0
    while True:
        jstat=job.status()
        elaT=time()-T0
        print('M:i=%d  status=%s, elaT=%.1f sec'%(i,jstat,elaT))
        if jstat=='SUCCESS': break
        if jstat=='ERROR': exit(99)
        i+=1; sleep(20)
    print('M: last circ=%d finished'%nCirc)

    #.... pull all results
    nCirc=expMD[ 'payload']['num_sample']
    resDL=[None]*nCirc
    for ic in range(nCirc):
        jid=sbm['job_ids'][ic]
        job = proc.resume_job(jid)
        assert job.status()=='SUCCESS'
        resDL[ic] = job.get_results()
        
    harvest_sampler_results(resDL,expMD,expD)
   
    if args.verb>2: pprint(expMD)
    
    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.meas.h5')
    write4_data_hdf5(expD,outF,expMD)


    print('   ./postproc_quandela.py  --expName   %s   -p a    -Y\n'%(expMD['short_name']))
  
    
    
