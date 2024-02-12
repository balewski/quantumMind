#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Part 6) Analyze yields from many QFT jobs
compute entropy for correct answer

Required INPUT:
   --jid6 list: j33ab44

'''
# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint
import numpy as np
from dateutil.parser import parse as date_parse

from OneExp_Ana import OneExp_Ana, import_yieldArray
from OneExp_AnaPlotter import OneExp_AnaPlotter

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for everything")
    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")
    parser.add_argument( "-X","--no-Xterm",dest='noXterm',action='store_true',
                         default=False, help="disable X-term for batch mode")
    parser.add_argument('-j',"--jid6", nargs='+',default=['jf02d1c'] ,help="shortened job ID list, blank separated ")

    args = parser.parse_args()
    args.prjName='anaQFT'
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
plot=OneExp_AnaPlotter(args)

''' structure of yieldArray data:
 - multiple Qiskit jobs, each has own yieldArray
 - 1 array comntains mutiple cycles of sequence of experiments
 Processing:
 - open all jid-yiled files and stack iformation for processing
 - a) sort experiments into the same type (done by AnaOne)
 - b) compute fidelity of each experiment
 - c) compute std dev of fidelity over many experiments of the same type
 - d) plot results
'''


print('read %s job yields'%len(args.jid6),args.prjName)

# (1) read 1st experimet to setup all variables
jid6=args.jid6[0]
metaD,yields=import_yieldArray(jid6,args.dataPath)

# (2)  prep exp names based on the last job

tmpS=set()
expMetaD= metaD['expInfo']['expMeta'] #assumes all jobs are identical
for x in expMetaD: tmpS.add ( expMetaD[x]['injGate']  ) 
expIdL=list(tmpS)
#print('expIdL:',expIdL)

# sort expId by : layer,gate_name,1st_qubit
expIdL.sort(key=lambda x: '%02d%s'%(int(x.split('.')[0]),x.split('.')[2]))
numExp=len(yields)
print('M:found %d different expId, each with %d experiments'%(len(expIdL),numExp))
print(', '.join(expIdL))

nclbit=len(metaD['expInfo']['meas_qubit'])
labels=[] # regenarte  them
for i in range(2**nclbit):
    #bitStr=bin(i)[2:].zfill(nclbit)
    bitStr='0x%0x'%i
    labels.append(bitStr)
print(1<<nclbit,'state_labels:',' '.join(labels))

oneD={} # key=expId
for expId in expIdL:
    # ... scan all yields and pick only those which match expId
    oneD[expId]=OneExp_Ana(expId,labels,metaD)

# (3) sum experiments for the same expId
totExp=0
for jid6 in args.jid6:
    print('injest jid6=',jid6)
    metaD,yields=import_yieldArray(jid6,args.dataPath)
    numExp==len(yields)  # all jobs must have the same shape
    totExp+=len(yields)
    execDate=date_parse(metaD['retrInfo']['exec_date'])
    for expId in expIdL:
        #if expId!='0.base.0' : continue #debug
        oneD[expId].unpack_yieldArray(jid6, execDate, yields)
print('\nM: imported %d jobs, totExp=%d, injPt=%d'%(len(args.jid6),totExp,len(oneD)))


baseExp='0.base.0'
oneD[baseExp].analyzeIt_perJob(verb=1)
baseData_perJob=oneD[baseExp].data_perJob

for expId in expIdL:
    oneD[expId].compensated_analysis()# baseData_perJob,verb=1 )
    
# - - - - - - - - 
# PLOT RESULTS
# - - - - - - - - 

#target='000'
target='0x0'
# track time stability 

one=oneD[baseExp]
plot.one_label_vs_date(one,target,obsN='counts',figId=10)
#one=oneD['8.cx.0.1']
#plot.one_label_vs_date(one,target,obsN='counts',figId=11)

# selected results
for expId in ['0.base.0'] :
    #break
    one=oneD[expId]
    #plot.histo_per_label(one,obsN='prob',figId=100)
    plot.histo_per_labelXX(one,obsN='prob',figId=100)
    #1plot.histo_per_label(one,obsN='selfInfo',figId=200)
    #plot.experiment_summary(one, obsN='selfInfo',figId=300)
    plot.experiment_summary(one, obsN='prob',figId=300)
    break

if len(expIdL)>1:
    plot.multi_experiment(oneD,target,expIdL, obsN='avr_selfInfo',figId=401)
    plot.multi_experiment(oneD,target,expIdL, obsN='avr_prob',figId=400)


plot.display_all('anaQFT', tight=True)
