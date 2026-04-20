#!/usr/bin/env python3
""" 
 concatenate mutiple measurements from Quera or from emulation
no graphics

Use case:
./aggregate_jobs.py --basePath /dataVault/dataQuEra_2023may1AtA_qpu --calTag rabiJune6
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import  time
import sys,os
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.Util_stats import do_yield_stats  # for heralding

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or env--> $QuEra_dataVault")

    parser.add_argument( "--calTag",  default='rabiJune6X',help=' calibration tag name')

    args = parser.parse_args()
    # make arguments  more flexible
    if 'env'==args.basePath: args.basePath= os.environ['QuEra_dataVault']
    args.dataPath=os.path.join(args.basePath,'ana')
    args.outPath=os.path.join(args.basePath,'calib')
  
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def locateSummaryData(src_dir,pattern=None,verb=1):

    for xx in [ src_dir]:
        if os.path.exists(xx): continue
        print('Aborting on start, missing  dir:',xx)
        exit(1)

    if verb>0: print('locate summary data src_dir:',src_dir,'pattern:',pattern)

    jobL=os.listdir(src_dir)
    print('locateSummaryData got %d potential jobs, e.g.:'%len(jobL), jobL[0])
    print('sub-dirs',jobL[:10])

    sumL=[]
    for sumF in jobL:
        if '.h5' not in sumF: continue
        if pattern not in sumF: continue
        sumL.append(os.path.join(src_dir,sumF))
        
    if verb>0: print('found %d sum-files'%len(sumL))
    return sorted(sumL)


#...!...!....................
def prep_metadata(md,dataD):
    import json
    pd=md['payload']
    
    # move few large items to big-data
    for key in ['program_org', 'atom_xy']:
        xD=pd.pop(key)
        xJ=json.dumps(xD, default=str) # Decimal --> str
        dataD[key+'.JSON']=xJ

    #... add new information
    qad=md['job_qa']
    qad['num_tbins']=dataD['evolTime_us'].shape[0]
    print('tt',type(qad['num_tbins']))
        
    return md
    
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()
     
    sumFL=locateSummaryData( args.dataPath,pattern='rabi')
    assert len(sumFL) >0
    if 0: # hack
        sumFL+=locateSummaryData( args.dataPath.replace("1AtA","1AtB"),pattern='rabi')
    
    evolTime=[]
    probsS=[]  # sum over all atoms
    heraldY=None
    for i,inpF in enumerate(sumFL):        
        expD,expMD=read4_data_hdf5(inpF,verb=i==0)
        if i==-1: pprint(expMD)
        
        tEvol=float(expMD['payload'].pop('evol_time_us'))
        evolTime.append(tEvol)

        probsS.append( expD['prob_sum'][:,0,:] )# skip measurement index
        if i==0:   # shape[ atom, NB]
            heraldY=expD['herald_counts_atom'].copy()
        else:
            heraldY+=expD['herald_counts_atom'] 
    # convert to numpy
    probsS=np.array(probsS) # shape[nTime, PAY,  NB]

    #print("a1",heraldY.shape)
    heraldP=do_yield_stats(heraldY)
    #print("a2",heraldP.shape)
    
    bigD={}
    bigD["evolTime_us"]=np.array(evolTime)
    bigD["probsSum"]=np.swapaxes(probsS,0,1) #  shape[ PAY, nTime, NB]
    bigD["herald_prob_atom"]=heraldP  # [PAY,atom,NB],  avr over all inputs
    print('M: evolTime/us:',evolTime,  bigD["probsSum"].shape)

    outMD=prep_metadata(expMD,bigD)
    outMD['short_name']=args.calTag
    pprint(outMD)
    print('bigD:',bigD.keys())
    
    
    #...... WRITE  OUTPUT
    outF=os.path.join(args.outPath,outMD['short_name']+'.ags.h5')
    #print('outF',outF); ok77
    write4_data_hdf5(bigD,outF,outMD)

    print('M:done')
