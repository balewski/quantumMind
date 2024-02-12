#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Toy study of impact of readout error and its mitigation for QuEra readout noise model:
* set ground truth probability for all possible bistrings of given lenght
* repeat mutiple times:
  * generate counts for all bistrings using nShots
  * (optional1) add decay of 1-state to 0-state
  * compute probabilities + std-dev
  * (optional2) unfold decayed counts & propagate error 
  * compute & accumulate chi2/dof vs. ground truth

* verify mean & std of chi2/dof

# use case:
./study_readErrMitt.py --numShot 500 --numRepeat  100 --numBit 2  --readErrEps 0.12    

'''

import time,os,sys
from pprint import pprint
import numpy as np
from bitstring import BitArray
from toolbox.Util_stats import do_yield_stats
from toolbox.Util_readErrMit import mitigate_probs_readErr_QuEra_1bit, mitigate_probs_readErr_QuEra_2bits, mitigate_probs_readErr_QuEra_3bits

class DataEmpty:    pass  # empty class
rng = np.random.default_rng()

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase output verbosity", default=1)
    parser.add_argument('--numBit', default=2, type=int, help='num of bits in measurement')
    parser.add_argument('-n','--numShot', default=23, type=int, help='num of shots')
    parser.add_argument('-r','--numRepeat', default=5, type=int, help='num of repetition of experiment')

    parser.add_argument('--readErrEps', default=None, type=float, help='probability of state 1 to be measured as state 0')
    

    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def set_true_probs(md):
    md.num_bit=args.numBit
    md.num_bitstr=1<<md.num_bit
    md.read_err_eps=args.readErrEps
    vect=rng.uniform(size=md.num_bitstr)
    vect/=np.sum(vect)
    
    #vect=np.array([0.2, 0.1, 0.3, 0.4])  # tmp
    return vect


#...!...!....................
def do_redchi(probT,dataP):  #.... compute chi2
    resV=(probT-dataP[0])/dataP[1]
    chi2=np.sum(resV**2)
    ndof=probT.shape[0]  #??? it should be '-1' because sum of probabilities must be 1.0
    redchi=chi2 /ndof
    if args.verb>1: print('  redchi==%.2f'%redchi)
    return redchi


#...!...!....................
def run_experiment(k,probT,md):

    # .... generate shots
    cntI=np.zeros((md.num_bitstr),dtype=np.int32) # ideal readout
    cntN=np.zeros((md.num_bitstr),dtype=np.int32) # noisy redout
    indices=np.arange(md.num_bitstr)
    for i in range(args.numShot):
        j= int(np.random.choice(indices, p=probT))
        cntI[j]+=1
        if md.read_err_eps!=None:
            # for each bit=1 try to flip it w/ prob=eps
            A=BitArray(uint=j,length=md.num_bit)
            bits=A.bin
            #print('A',j,bits)
            out=''
            xx=rng.uniform()
            for ib in bits:
                if ib=='1':  # try to decay bit 1
                    if rng.uniform()<md.read_err_eps: ib='0'                    
                out+=ib
            #print('in:',bits, 'out:',out)
            B=BitArray(bin=out)
            #print('B',B.uint)
            cntN[B.uint]+=1
    if args.verb>1:
        print('  cntI:',cntI, 'sum=',np.sum(cntI))
        print('  cntN:',cntN, 'sum=',np.sum(cntN))
        
    # ...  prob. for ideal readout
    dataYI=cntI.reshape(1,-1)
    dataPI=do_yield_stats(dataYI,verb=0)[:,0,:]
    if args.verb>1: print('  rec probsI:\n',dataPI)
    redchiI=do_redchi(probT,dataPI)

    redchiN=-3.3
    if md.read_err_eps!=None:
        # ... prob. for noisy readout
        dataYN=cntN.reshape(1,-1)
        dataPN=do_yield_stats(dataYN,verb=0)
        redchiN=do_redchi(probT,dataPN[:,0,:])

        # ... prob. for mitigated noisy readout
    
        if md.num_bit==1:
            dataPM=mitigate_probs_readErr_QuEra_1bit(dataPN,md.read_err_eps)
        elif md.num_bit==2:
            dataPM=mitigate_probs_readErr_QuEra_2bits(dataPN,md.read_err_eps)
        elif md.num_bit==3:
            dataPM=mitigate_probs_readErr_QuEra_3bits(dataPN,md.read_err_eps)
        else:
            missing_mitigate_probs_Xbits
            
        redchiM=do_redchi(probT,dataPM[:,0,:])
    else:
        redchiM=-2.2
        
    return redchiI, redchiN, redchiM

        
        
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    np.set_printoptions(precision=3)
    meta=DataEmpty()
    trueProb=set_true_probs(meta)
    
    print('M: true prob:',trueProb,'sum=',np.sum(trueProb))
    
    chiIL=[];chiNL=[];chiML=[]
    for k in range(args.numRepeat):
        if args.verb>1 or k%100==0: print('  L: repeat=',k)
        chiI,chiN,chiM=run_experiment(k,trueProb,meta)
        chiIL.append(chiI)
        chiNL.append(chiN)
        chiML.append(chiM)

    chiIL=np.array(chiIL);     chiNL=np.array(chiNL);     chiML=np.array(chiML)
    avrChiI=np.mean(chiIL);    stdChiI=np.std(chiIL)
    avrChiN=np.mean(chiNL);    stdChiN=np.std(chiNL)
    avrChiM=np.mean(chiML);    stdChiM=np.std(chiML)

    print('\nM: true prob:',trueProb,'sum=',np.sum(trueProb))
    print('\nM: ideal avr chi2/dof=%.2f +/- %.2f , num repeat=%d'%(avrChiI,stdChiI,args.numRepeat))
    print( 'M: noisy avr chi2/dof=%.2f +/- %.2f '%(avrChiN,stdChiN))
    print( 'M: mittg avr chi2/dof=%.2f +/- %.2f '%(avrChiM,stdChiM))
    print('DONE')
