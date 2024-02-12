#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 Retrieve  results of IBMQ job

Required INPUT:
    --expName: exp_j33ab44

Output:  raw  yields + meta data
'''

import time,os,sys
from pprint import pprint
import numpy as np
from toolbox.Util_H5io4 import  read4_data_hdf5, write4_data_hdf5
from toolbox.UQiskit_job import access_ibm_provider
 
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase output verbosity", default=1)
    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or env--> $QCrank_dataVault")
    parser.add_argument('-e',"--expName",  default='exp_62a15b79',help='IBMQ experiment name assigned during submission')

    args = parser.parse_args()
    if 'env'==args.basePath: args.basePath= os.environ['QCloud_dataVault']
   
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

    inpF=args.expName+'.ibmq.h5'
    expD,jobMD=read4_data_hdf5(os.path.join(args.inpPath,inpF),verb=args.verb)
     
    if args.verb>1: pprint(jobMD)

    if 0:    #example decode one Qasm2 circuit
        #rec2=expD['circ_qasm3'][1].decode("utf-8")
        rec2=expD['transp_qasm3'][1].decode("utf-8") 
        print('\nM:qasm3 circ1:\n',rec2)

    # select Task-class:
    if 'TaskTomoU3'==jobMD['payload']['class_name']:
        from TaskTomoU3      import TaskTomoU3
        task=TaskTomoU3(args,jobMD,expD)
    if 'TaskMidCircMeas1Q'==jobMD['payload']['class_name']:
        from TaskMidCircMeas1Q import TaskMidCircMeas1Q
        task=TaskMidCircMeas1Q(args,jobMD,expD)

    
    provider = access_ibm_provider(jobMD['submit']['provider'])
    
    task.retrieveIBM_job(provider)
   
    if args.verb>2: pprint(expMD)
    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,jobMD['short_name']+'.h5')
    write4_data_hdf5(expD,outF,jobMD)

    #..........  helper analysis  programs

    anaCode= jobMD['analyzis']['ana_code']
    print('\n   ./%s --expName   %s    \n'%(anaCode,jobMD['short_name']))  
    
       


