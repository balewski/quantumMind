#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
merge yields from many Z2Phase jobs as one hd5

'''

import os
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
import json
from pprint import pprint
import numpy as np
from collections import Counter

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or env--> $QPIXL_dataVault")
                        
    parser.add_argument('-e',"--expName",  default=['zurek_qpu_At2.1o58c','zurek_qpu_At2.1o58b'],  nargs='+',help='list of retrieved experiments, blank separated')

    args = parser.parse_args()
    # make arguments  more flexible
    if 'env'==args.basePath: args.basePath= os.environ['QuEra_dataVault']
    args.dataPath=os.path.join(args.basePath,'meas')
 
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.dataPath)
    assert len(args.expName) == len(set(args.expName))  # check for duplicates
    return args

#...!...!.................... 
def add_experiment(ie,outD,outMD):
    inpF=args.expName[ie]+'.h5'
    print('A: %d %s'%(ie,inpF))
    expD1,expMD1=read4_data_hdf5(os.path.join(args.dataPath,inpF),verb=0)

    #... extract bigData
    countsRaw=json.loads(expD1.pop('counts_raw.JSON')[0])

    #.... minimal sanity check
    assert str(outMD['submit']['backend'])==str(expMD1['submit']['backend'])
    assert expMD1['job_qa']['success']
    
    #pprint(expMD['job_qa'])
    # update merge=MD
    mrd=outMD['merge']
    x='hash'; mrd[x].append(expMD1[x])
    x='short_name'; mrd[x].append(expMD1[x])
    x='exec_date';  mrd[x].append(expMD1['job_qa'][x])
    x='num_sol' ;   mrd[x].append(expMD1['job_qa'][x])
    x='num_shots' ; mrd[x].append(expMD1['submit'][x])
    return countsRaw         
         
#...!...!.................... 
def setup_containers(expD,expMD,numJobs):  # overwrites both containers
    # experiments may have different number of shots
    print('SC: MD keys:',sorted(expMD))
    print('SC: bigD keys:',sorted(expD))

    bigD=expD
    bigD.pop('shots_raw.JSON') # discard
    countsRaw=json.loads(bigD.pop('counts_raw.JSON')[0])
        
    print('created bigD:',list(bigD))    
    #2 ... meta data...              
    assert expMD['job_qa']['success']

    MD=expMD
    
    #.... add merging input info
    mrd={'num_jobs':numJobs}
    MD['merge']=mrd
    mrd['hash']=[ expMD['hash']]        
    mrd['short_name']=[ expMD['short_name']]  #  will be changed below       
    x='exec_date';  mrd[x]=[expMD['job_qa'][x]]
    x='num_sol' ;   mrd[x]=[expMD['job_qa'].pop(x)]  # after heralding
    x='num_shots' ; mrd[x]=[expMD['submit'].pop(x)]  # as requested
      
    # ... new merged job hash & name
    MD['hash']='%sM%d'%(expMD['hash'],numJobs)
    MD['short_name']='%sM%d'%(expMD['short_name'],numJobs)

    #pprint(MD)
    return bigD, MD,countsRaw

#...!...!....................
def merge_and_sum_dictionaries(dict_list):
    result_counter = Counter()
    for d in dict_list:
        result_counter.update(d)
    return dict(result_counter)
    '''
To optimize the merging and summing of very large dictionaries, especially when the key overlap is significant, we can utilize collections.Counter. This class is designed for counting hashable objects and is an ideal fit for this use case. It inherently manages adding counts for keys, simplifying and potentially speeding up the operation for large datasets.
    '''
                        
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()

    numJobs=len(args.expName)
    inpF=args.expName[0]+'.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))

    if args.verb>1: pprint(expMD)

    #... holder for packed bitstrings
    countsLL=[ None for _ in range(numJobs) ]
    outD,outMD,countsL=setup_containers(expD,expMD,numJobs)
    countsLL[0]=countsL
    print('zz0',type(countsL))
    #pprint(countsL)
    
    # append other experiments
    for ie in range(1,numJobs):
        countsLL[ie]=add_experiment(ie,outD,outMD)
        
    print('M:merge info');pprint(outMD['merge'])

    #.... merge bitstrings
    nSolL=outMD['merge']['num_sol']
    nSolSum=sum(nSolL)
    print('M: sum nSol:',nSolSum)
    countsAll=merge_and_sum_dictionaries(countsLL)
    total_sum = sum(countsAll.values())
    print('M:countsAll: true nSol=%d  sumCounts=%d'%(len(countsAll),total_sum))

    outD['counts_raw.JSON']=json.dumps(countsAll)  #  dictionary of egr-strings counts
    outMD['job_qa']['num_sol']=len(countsAll)
    outMD['submit']['num_shots']=total_sum

    if args.verb>1: pprint(outMD)
    #...... WRITE  OUTPUT .........

    outF=os.path.join(args.dataPath,outMD['short_name']+'.h5')
    print('M: merge outF=',outF)
    assert os.path.exists(outF)==False
    
    write4_data_hdf5(outD,outF,outMD)
    
    print('   ./postproc_Z2Phase.py  --basePath  %s  --expName %s  -p r s e  d c m  f -X    \n'%(args.basePath,outMD['short_name'] ))

