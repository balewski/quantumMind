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
#from OneExp_AnaPlotter import OneExp_AnaPlotter

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
    args.prjName='anaScanT1'
    args.dataPath+='/' 
    args.outPath+='/' 
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def XXnice_table(sumAna,shots):
    print('\nnice table, prep1, meas0')
    for k in sumAna:
        delayT,prepState=k
        cnt=sumAna[k]
        if prepState==0: continue
        #print(delayT,cnt)
        avr,std,num,err=cnt[0]
        avr/=shots
        err/=shots
        if delayT==0:
            bPr=avr; bPrEr=err
            print('delT=0  prob=%.3f +/- %.3f'%(avr,err))
            continue
        else:
            avr-=bPr
            err=np.sqrt(err*err + bPrEr*err)
        T1=delayT/avr
        T1er=T1*err/avr
        print('delT=%d usec,  delProb=%.3f +/- %.3f,  T1=%.1f +/- %.1f'%(delayT,avr,err,T1,T1er))
#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()
#plot=OneExp_AnaPlotter(args)

print('M: read %s job yields'%len(args.jid6),args.prjName)

# (1) read 1st experimet to setup all variables
jid6=args.jid6[0]

metaD,yields=import_yieldArray(jid6,args.dataPath)
pprint(metaD)
numCirc=len(yields)
shots=metaD['expInfo']['shots']
expNameL=metaD['expInfo']['exp_name']
delayList=metaD['expInfo']['delayT_usec']

# (2)  prep labels 
qid=metaD['expInfo']['meas_qubit'][0]
print('M: Q=%d, found %d different experiments, total %d circ, jid6=%s'%(qid,len(expNameL),numCirc,jid6))


oneD={} # key=expId
for expId in expNameL:
    # ... scan all yields and pick only those which match expId
    oneD[expId]=OneExp_Ana(expId,metaD)
# prep is done, now read all data from all jobs

# (3) sum experiments for the same expId
totExp=0
for jid6 in args.jid6:
    print('injest jid6=',jid6)
    metaD,yields=import_yieldArray(jid6,args.dataPath)
    numCirc==len(yields)  # all jobs must have the same shape
    totExp+=len(yields)
    execDate=date_parse(metaD['retrInfo']['exec_date'])
    assert shots==metaD['expInfo']['shots']
    for expId in expNameL:
        oneD[expId].unpack_yieldArray(jid6, execDate, yields)
print('\nM: imported %d jobs, totExp=%d, injPt=%d'%(len(args.jid6),totExp,len(oneD)))


anaSum={}

for delayT in delayList:    
    name='scan_Q%d_us%d'%(qid,delayT)
    for prepState in [0,1]:
        expId=name+'_pr%d'%prepState
        #print(name,'extract meas for prepBit=',prepState)
        cnt=oneD[expId].singleBit_counts(verb=1)
        anaSum[(delayT,prepState)]=cnt
        print(expId,cnt)
    #break
print('M: anaSum [prep]:cnt[meas=0: (avr,std,num,err), m=1:...] ')

nice_table(anaSum,shots)

ok89
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
