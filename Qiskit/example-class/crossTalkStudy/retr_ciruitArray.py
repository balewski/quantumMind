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

sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, circuit_summary, write_yaml, read_yaml, retrieve_job, read_submitInfo


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
def extract_yields(jobResD,numMeasLab):
    # must count from 0 and all must exist
    measLabHex=[ '0x%x'%i for i in range(numMeasLab)]
    measLabInt=[ i for i in range(numMeasLab)]
    print('expect measLab',measLabHex)
    outL=[]
    for i, exp in enumerate(jobResD):
        expName=exp['header']['name']
        countA=[ 0 for i in range(numMeasLab) ]
        countD=exp['data']['counts']
        sTot=0
        for i in range(numMeasLab):
            try:
                countA[i]=countD[measLabHex[i]]
                sTot+=countA[i]
            except:
                continue
        outL.append( { 'counts':countA , 'name':expName})
        assert sTot>0 # if nothing is found the measLabels are worng
    return outL,measLabInt
    #print('rrr'); pprint(outL)

    
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

num_clbit=len(args.metaD['expInfo']['meas_qubit'])
numMeasLab=1<< num_clbit
args.metaD['yields'],measLabInt=extract_yields(jobResD,numMeasLab)

# add more from retrieve here
args.metaD['retrInfo']={'exec_duration':jobHeadD['time_taken'],'exec_date':jobHeadD['date'], 'numMeasLab':numMeasLab, 'measLabInt':measLabInt } 

outF=args.dataPath+'/yieldArray-%s.yaml'%args.jid6
write_yaml(args.metaD,outF)

numExp=len(jobResD)

for iexp in range(numExp):
    raw_counts = jobResD[iexp]['data']['counts']
    expName=jobResD[iexp]['header']['name']
    print('iexp=',iexp,'shots=',args.metaD['expInfo']['shots'],' name=',expName)
    print('raw',raw_counts)
    proc_counts=args.metaD['yields'][iexp]
    print('proc',proc_counts)
    if iexp>=0:break

print('End-OK')
