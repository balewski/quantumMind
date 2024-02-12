#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Utility tool: retrieve any jobs, print usefull info

Required INPUT:
   --job_id: 5d67...09dd  OR --jid6 j33ab44

'''

import time,os,sys
from pprint import pprint

from NoiseStudy_Util import read_submitInfo
sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, circuit_summary, read_yaml, retrieve_job

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for everything")

    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")
    parser.add_argument( "-X","--no-Xterm",dest='noXterm',action='store_true',
                         default=False, help="disable X-term for batch mode")
    parser.add_argument('-J',"--job_id",default=None ,help="full Qiskit job ID")
    parser.add_argument('-j',"--jid6",default='je840f5' ,help="shortened job ID")

    args = parser.parse_args()
    args.prjName='retrAny'    
    args.dataPath+='/'
    args.outPath+='/' 
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()
read_submitInfo(args)

backName=args.metaD['submInfo']['backName']

backend=access_backend(backName)
job, jobHeadD, jobResD = retrieve_job(backend, args.job_id, verb=args.verb)

print('jobRes header:'); pprint(jobHeadD)
assert jobHeadD['success']

numExp=len(jobResD)

print('job found, num experiments=',numExp,', status=',jobHeadD['status'])
assert numExp>0

iexp=numExp//2
exp1=jobResD[iexp]
print('example experiment i=',iexp); pprint(exp1)
shots=exp1['shots']
runDate=jobHeadD['date']
elaTime=jobHeadD['time_taken']
print(runDate,' elapsed time:%.1f sec, shots=%d'%(elaTime,shots))

