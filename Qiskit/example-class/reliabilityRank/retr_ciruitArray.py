#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

Part 5) retrieve  target circuit results 

Required INPUT:
   --job_id: 5d67...09dd  OR --jid6 j33ab44
   --transpName : ghz_5qm.v1

???Reads matching  transp-file, recovers backend name
  data/transp.ghz_5qm.v10.yaml

Output:  raw  yields + meta data

    args.prjName='measCirc_'+args.transpName

'''
# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint

from qiskit.ignis.mitigation.measurement import  CompleteMeasFitter
sys.path.append(os.path.abspath("../noiseStudy/"))
from NoiseStudy_Util import read_submitInfo
sys.path.append(os.path.abspath("../../utils/"))

from Circ_Plotter import Circ_Plotter
from Circ_Util import access_backend, circuit_summary, write_yaml, read_yaml, print_measFitter,  save_measFitter_Yaml, restore_measFitter_Yaml, retrieve_job

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
def yields_2_metaD(args,jobResMitgD,expMetaD):
    outL=[]
    for exp in jobResMitgD:
        #print('gg'); pprint(exp); ok34
        counts=exp['data']['counts']
        counts2={ x:float(counts[x] ) for x in counts} # get rid of numpy
        expName=exp['header']['name']
        injGate=expMetaD[expName]['injGate']
        outL.append( { 'shots': exp['shots'], 'counts':counts2  , 'name':expName,'injGate':injGate})
    args.metaD['yieldsMitg']=outL
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

numExp=len(jobResD)
    
yields_2_metaD(args,jobResD,args.metaD['expInfo']['expMeta'])
# add more from retrieve here
metaD=args.metaD
args.metaD['retrInfo']={'exec_duration':jobHeadD['time_taken'],'exec_date':jobHeadD['date'] } 

outF=args.dataPath+'/yieldArray-%s.yaml'%args.jid6
write_yaml(args.metaD,outF)

for iexp in range(numExp):
    raw_counts = jobResD[iexp]['data']['counts']
    print('iexp=',iexp,'shots=',args.metaD['expInfo']['shots'])
    print('raw',raw_counts)
    break

print('End-OK')
