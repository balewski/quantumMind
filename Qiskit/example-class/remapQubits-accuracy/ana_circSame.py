#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 Analyze yields from  one scanT1 job

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
    parser.add_argument('-p',"--probRange", nargs='+',type=float,default=[0.,1.01],
                        help="[min, max] probability for plotting ")

    
    args = parser.parse_args()
    args.prjName='anaScanT1'
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
#args.probRange=[0.81,0.99]
plot=OneExp_AnaPlotter(args)

print('M: read %s job yields'%len(args.jid6),args.prjName)

# (1) read 1st experimet to setup all variables
jid6=args.jid6[0]

metaD,yields=import_yieldArray(jid6,args.dataPath)
pprint(metaD)
numCirc=len(yields)
shots=metaD['expInfo']['shots']
expConf=metaD['expInfo']['conf']
expNameL=list(expConf.keys())

# (2)  prep labels 
#qid=metaD['expInfo']['meas_qubit']
print('M:  found %d different experiment(s), total %d circ, jid6=%s'%(len(expNameL),numCirc,jid6))

oneD={} # key=expId
for expId in expNameL:
    # ... scan all yields and pick only those which match expId
    oneD[expId]=OneExp_Ana(expId,metaD)
# prep is done, now read all data from all jobs

# (3) sum experiments for the same expId
totCirc=0
for jid6 in args.jid6:
    print('injest jid6=',jid6)
    metaD,yields=import_yieldArray(jid6,args.dataPath)
    numCirc==len(yields)  # all jobs must have the same shape
    totCirc+=len(yields)
    execDate=date_parse(metaD['retrInfo']['exec_date'])
    assert shots==metaD['expInfo']['shots']
    for expId in expNameL:
        oneD[expId].unpack_yieldArray(jid6, execDate, yields)
print('\nM: imported %d jobs, totCirc=%d, numExp=%d'%(len(args.jid6),totCirc,len(oneD)))

for expId in oneD:
    one=oneD[expId]
    one.doAverage()
    #print(expId,cnt)
    plot.histo_per_label(one,figId=100)
    plot.experiment_summary(one,figId=200)
    #break

plot.display_all('anaQFT', tight=True)

