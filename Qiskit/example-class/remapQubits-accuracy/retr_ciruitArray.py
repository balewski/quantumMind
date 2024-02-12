#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 retrieve  target circuit results 

Required INPUT:
   --job_id: 5d67...09dd  OR --jid6 j33ab44

Output:  raw  yields + meta data

'''
# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint
import numpy as np

sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, circuit_summary, write_yaml, read_yaml, retrieve_job, read_submitInfo
from  ReadErrCorrTool  import ReadErrCorrTool

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for calib+circ")
    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")
    parser.add_argument( "-X","--no-Xterm",dest='noXterm',action='store_true',
                         default=False, help="disable X-term for batch mode")

    parser.add_argument('-J',"--job_id",default=None ,help="full Qiskit job ID")
    parser.add_argument('-j',"--jid6",default='je840f5' ,help="shortened job ID")

    args = parser.parse_args()
    
    args.dataPath+='/' 
    args.outPath+='/' 
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def extract_yields(jobResD,nclbit,hasRdErrCorr):
    print('EY: num circ=%d hasRdErrCorr=%s'%(len(jobResD),hasRdErrCorr))
    # must count from 0 and all must exist
    numMeasLab=1<<nclbit
    measLabBin=[ bin(i)[2:].zfill(nclbit) for i in range(numMeasLab)]
    measLabHex=[ '0x%x'%i for i in range(numMeasLab)]
    measLabInt=[ i for i in range(numMeasLab)]
    print('expect %d measLab'%numMeasLab,measLabBin)
    outL=[]
    for j, exp in enumerate(jobResD):
        expName=exp['header']['name']
        countA=[ 0. for i in range(numMeasLab) ]
        countD=exp['data']['counts']
        sTot=0
        for i in range(numMeasLab):
            try:
                if hasRdErrCorr:
                    countA[i]=countD[measLabBin[i]]
                else:
                    countA[i]=countD[measLabHex[i]]
                sTot+=countA[i]
            except:
                continue
        countA=np.array(countA)
        #print('qqq',countA.shape)#,type(countA))
        outL.append( { 'counts':countA , 'name':expName})
        assert sTot>0 # if nothing is found the measLabels are worng
    return outL,measLabInt


    
#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()
read_submitInfo(args)
backName=args.metaD['submInfo']['backName']
backend=access_backend(backName)
job, jobHeadD, jobResL = retrieve_job(backend, args.job_id, verb=args.verb)

hasRdErrCorr=False
if 'rdCorInfo' in args.metaD:
    print('read-error-correction will be applied to experiments')
    hasRdErrCorr=True
    rdCorConf=args.metaD['rdCorInfo']
    rdCor=ReadErrCorrTool(rdCorConf)
    rdCor.compute_corrections(job)
    jobMResL=rdCor.apply_corrections(job)
    print('aa',len(jobResL),len(jobMResL))
    jobResL=jobMResL

assert len(jobResL)>0
#num_clbit=len(rdCor.conf['meas_qubit'])
num_clbit=jobResL[0]['header']['creg_sizes'][0][1]

#exp=jobResD[0]
#counts=exp['data']['counts']
#print('M',counts,)

args.metaD['yields'],measLabInt=extract_yields(jobResL,num_clbit,hasRdErrCorr)

# add more from retrieve here:
args.metaD['retrInfo']={'exec_duration':jobHeadD['time_taken'],'exec_date':jobHeadD['date'], 'measLabInt':measLabInt,'rdErrCorr':hasRdErrCorr } 

outF=args.dataPath+'/yieldArray-%s.yaml'%args.jid6
write_yaml(args.metaD,outF)

numExp=len(jobResL)

for iexp in range(numExp):
    counts = jobResL[iexp]['data']['counts']
    expName=jobResL[iexp]['header']['name']
    print('iexp=',iexp,'shots=',args.metaD['expInfo']['shots'],' name=',expName)
    print('counts',counts)
    proc_counts=args.metaD['yields'][iexp]
    print('the same',proc_counts)
    if iexp>=0:break

print('End-OK')
