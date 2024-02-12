#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Part 6) Analyze yields from many Grover jobs

Required INPUT:
   --jid6 list: j33ab44

'''
# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint
import numpy as np

from CircGrov3Q_AnaOne import CircGrov3Q_AnaOne
from CircGrov3Q_Plotter import  CircGrov3Q_Plotter
from Circ_Util import  write_yaml, read_yaml

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
    args.prjName='anaGrov3Q'
    args.dataPath+='/' 
    args.outPath+='/' 
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def import_yieldArray(jid6):
        inpF=args.dataPath+'yieldArray-%s.yaml'%jid6
        blob=read_yaml(inpF)
        print('blob keys:', blob.keys())
        yields=blob.pop('yieldsMitg')
        metaD=blob
        print('metaD:', metaD.keys())
        print('see %d experiments'%len(yields))
        return metaD,yields
 
#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()
plot=CircGrov3Q_Plotter(args)
''' structure of yieldArray data:
 - multiple Qiskit jobs, each has own yieldArray
 - 1 array comntains mutiple cycles of sequence of experiments
 Processing:
 - a) sort experiments into the same type (done by AnaOne)
 - b) compute fidelity of each experiment
 - c) compute std dev of fidelity over many experiments of the same type
 - d) plot results
'''

print('read %s job yields'%len(args.jid6),args.prjName)
jobDataL=[]
totExp=0
baseL=[]
for jid6 in args.jid6:
   print('injest jid6=',jid6)
   metaD,yields=import_yieldArray(jid6)
   expId=0
   one=CircGrov3Q_AnaOne(expId,[[jid6,metaD,yields]])
   one.analyzeIt()
   baseL.append(one) # to get average base result per job
   jobDataL.append([jid6,metaD,yields])
   totExp+=len(yields)
print('M: imported %d jobs , totExp=%d'%(len(jobDataL),totExp))


#prep exp names based on last experiment
tmpS=set()
expMetaD= metaD['expInfo']['expMeta']
for x in expMetaD: tmpS.add ( expMetaD[x]['injGate']  ) 
expIdL=list(tmpS)
print('expIdL:',expIdL)

expIdL.sort(key=lambda x: '%02d%s'%(int(x.split('.')[0]),x.split('.')[2]))
print('M:found %d different experiments'%len(expIdL))
print(', '.join(expIdL))

pmax=0.9+.11
# seed one-objects with data
oneD={}
for expId in expIdL:
   one=CircGrov3Q_AnaOne(expId,jobDataL)
   one.analyzeIt()
   oneD[expId]=one
   if expId not in ['0.base.0','4.cx.2.1','7.cx.2.1'] : continue
   
   plot.states_prob_histo(one, pmax=pmax,figId=100)
   plot.experiment_summary(one, pmax=pmax,figId=200)
  
print('captured %d experiments'%len(oneD))
plot.muti_experiment(oneD,'101',expIdL,figId=300)
   
plot.display_all('anaGrov3Q', tight=True)
