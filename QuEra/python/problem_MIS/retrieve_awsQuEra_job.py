#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

 Retrieve  QuEra results of AWS job

Required INPUT:
    --expName: exp_j33ab44

Output:  raw  bit-strings + meta data  saved as HD5

'''

import time,os,sys
from pprint import pprint
import numpy as np
from toolbox.Util_H5io4 import  read4_data_hdf5, write4_data_hdf5
from toolbox.UAwsQuEra_job import  access_quera_device
from ProblemMaxIndependentSet import ProblemMIS
        
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase output verbosity", default=1)
    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or env--> $QuEra_dataVault")
    parser.add_argument('-e',"--expName",  default='exp_62a15b',help='AWS-QuEra experiment name assigned during submission')       

    args = parser.parse_args()
    if 'env'==args.basePath: args.basePath= os.environ['QuEra_dataVault']
   
    args.inpPath=os.path.join(args.basePath,'jobs')
    args.outPath=os.path.join(args.basePath,'meas')
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


def patch_9880ab(jobMD):
    # use regenerated job submission data/meta
    expDx,jobMDx=read4_data_hdf5('/dataVault/dataQuEra_2023june/jobs/fmis_8771c8.quera.h5')
    #pprint(jobMDx)
    for name in ['short_name','submit']:
        jobMDx[name]=jobMD[name]
    return expDx,jobMDx

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    inpF=args.expName+'.quera.h5'
    expD,jobMD=read4_data_hdf5(os.path.join(args.inpPath,inpF),verb=args.verb)

    #expD,jobMD=patch_9880ab(jobMD) # fix june17 data
    
    if args.verb>1: pprint(jobMD)
   
    if 0: # tmp, Thursday
        #jobMD['payload']['class_name']='ProblemAtomGridSystem'
        expMD=jobMD
        expMD['payload']['num_clust']=1
        expMD['payload']['num_atom_in_clust']=7
        expD['hamiltonian.JSON']=expD.pop('program_org.JSON')

              
    task=ProblemMIS(args,jobMD,expD)

    task.retrieve_job() 
   
    if args.verb>2: pprint(expMD)
    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,jobMD['short_name']+'.h5')
    write4_data_hdf5(expD,outF,jobMD)

    #..........  helper analysis  programs
    anaCode= jobMD['analyzis']['ana_code']
    baseStr= " --basePath %s "%args.basePath if args.basePath!=os.environ['QuEra_dataVault']  else ""  # for next step program
    print('\n   ./%s %s --expName   %s  -p r d s  -X \n'%(anaCode,baseStr,jobMD['short_name']))
    
       


