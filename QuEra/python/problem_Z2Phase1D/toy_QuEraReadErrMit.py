#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
simulate readout error and its mitigation

./toy_QuEraReadErrMit.py --data_bin 00101100 11110000 00100110  

Truth:
u:ii 0 hex=2c hw=3 mshot=1000
u:ii 1 hex=f0 hw=4 mshot=1500
u:ii 2 hex=26 hw=3 mshot=2000

noisy 
{'02': 11,
 '04': 15,
 '06': 138,
 '08': 7,
 '0c': 73,
 '10': 1,
 '20': 16,
 '22': 151,
 '24': 250,
 '26': 1509,  WAS 2000
 '28': 69,
 '2c': 763,   WAS 1000
 '30': 7,
 '40': 1,
 '50': 7,
 '60': 10,
 '70': 92,
 '90': 6,
 'a0': 13,
 'b0': 84,
 'c0': 8,
 'd0': 94,
 'e0': 84,
 'f0': 1091}   WAS 1500


mitigated  4500 shots:
{'02': 11,
 '04': 15,
 '06': 0,
 '08': 7,
 '0c': 0,
 '10': 1,
 '20': 16,
 '22': 0,
 '24': 43,
 '26': 1936,  WAS 2000
 '28': 0,
 '2c': 974,  WAS 1000
 '30': 0,
 '40': 1,
 '50': 0,
 '60': 0,
 '70': 29,
 '90': 6,
 'a0': 10,
 'b0': 0,
 'c0': 0,
 'd0': 0,
 'e0': 4,
 'f0': 1447}  WAS 1500

no graphics
'''

import os
from pprint import pprint
import numpy as np
import json 
from bitstring import BitArray
class EmptyClass:  pass
from collections import Counter

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3,4],  help="increase output verbosity", default=1, dest='verb')
         
    parser.add_argument("--data_bin",  default=['00101100'], nargs='+', help='ground truth binary string')
    parser.add_argument('-S',"--rnd_seed",  default=33, type=int,help='numpy random seed, 0=time()')


    args = parser.parse_args()
    # make arguments  more flexible
    args.nAtom=len(args.data_bin[0])
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def generateReadError(task,readErrEps=0.08):
        amd=task.meta['analyzis']
        amd['add_readErr_eps']=readErrEps
        nAtom=amd['num_atom']
        
        #... access true data
        dataY,pattV,hammV=task.dataTrue
        cntD={}
        
        def spread_measurements(patt,mshot):
            A=BitArray(hex=patt) # keep leading 0s to retain hex-capability
            oneIdxL=[  i  for i, bit in enumerate(A) if  bit]
            #print('parent=',hexpatt,A.bin, 'mshot=',mshot, 'oneIdxL:',oneIdxL)

            #mshot=100   
            for j in range(mshot):
                C=A.copy()
                #np.random.shuffle(oneIdxL)  # maybe over-kill?
                for i in oneIdxL:  # conditional bit-1 flip
                    if np.random.uniform()< readErrEps: C[i]=0                    
                patt=C.hex
                #print(j,A,'C=',C.bin,'k=',k)
                if patt not in cntD: cntD[patt]=0
                cntD[patt]+=1  # add one shot

        for hexpatt,mshot in zip( pattV,dataY):
            #if hexpatt not in accPattL: continue
            spread_measurements(hexpatt,mshot)
                        
            
        print('cntD nSol=%d'%(len(cntD))); print(cntD)
        return cntD
        

         
#...!...!....................
def dict_2_expD(cntD):  # 
        nSol=len(cntD)
        print('UED cntD:');pprint(cntD)
        
        #...... update meta-data
        #amd['num_sol']=nSol
        #rankedSol=solCounter.most_common(nSol)
        
        NB=nSol # number of bitstrings,  this is sparse encoding 
        dataY=np.zeros(NB,dtype=np.int32)  # measured shots per pattern
        pattV=np.empty((NB), dtype='object')  # meas bitstrings as hex-strings (hexpatt)
        hammV=np.zeros(NB,dtype=np.int32)  # hamming weight of a pattern

        print('dataY:',dataY.shape)
        i=0
        for hexpatt,mshot in cntD.items():
            A=BitArray(hex=hexpatt) # keep leading 0s
            hw=A.bin.count('1')  # Hamming weight
            if i<100: print('u:ii',i,'hex=%s %s hw=%d mshot=%d'%(A.hex,A.bin,hw,mshot))
            dataY[i]=mshot
            pattV[i]=A.hex
            hammV[i]=hw
            i+=1
            
        if 1:
            print('u:dataY:',dataY)
            print('u:pattV',pattV)
            print('u:hammV',hammV)

        return dataY,pattV,hammV
   
#...!...!....................
def split_by_Hamming_weight(XXX):
        dataY,pattV,hammV=XXX
        outD={}
        mxhw=0
        for mshot,patt,hw in zip(dataY,pattV,hammV):
            #print('solId:',solId,hw,mshot)
            if hw not in outD: outD[hw]={}
            outD[hw][patt]=mshot
            if mxhw<hw: mxhw=hw
        outD[mxhw+1]={}  # placeholder for promotion
        return outD
    
#...!...!....................
def mitigateReadErr(inpDD,task):

        minHw=task.meta['readErrMit']['min_hamm_weight']  # minimal corrected HW
        hwL=sorted(inpDD)[:-1]  # skip promoted hw
        print('MRE:hwL',hwL)

        for hw in hwL:  # hw needs to increase to correct counts for the next layer
            if hw<minHw: continue  # do not deal w/ low HW strings
            print('\nreduce counts for measurements with hw=',hw)
       
            deplete_hw_patterns(inpDD[hw],task.meta,inpDD[hw+1])

        print('MRE end:',inpDD)            

        #.... flatten 2D dict
        pprint(inpDD)
        outD={}
        tShot=0
        for  hw in inpDD:
            for patt,mshot in inpDD[hw].items():
                #print('kk',patt,sorted(outD))
                assert patt not in outD
                outD[patt]=mshot
                tShot+=mshot
        print('mitigated  %d shots:'%tShot);pprint(outD)
        return outD
                
            
#...!...!....................
def common_parent(cliqueS,nAtom): # find common parent for a clique
    D={}
    for patt in cliqueS:
        A=BitArray(hex=patt)
        print('CMP:child/hex:',patt,'bin:',A.bin,nAtom)        
        # try to replace every 0 -->1, one at a time
        for i in range(nAtom):  # take LSB so counts form the other end
            if A[-i] : continue
            C=A.copy()
            C[-i]=1 # un-flip  i-th bit          
            pattp=C.hex  # parent candidate
            #print('i=%d A=%s p=%s'%(i,A.bin,C.bin))
            if pattp not in D: D[pattp]=[]
            D[pattp].append(patt)
    print('CMP: D',D)
    pattp= max(D, key=lambda k: len(D[k]))  # key_with_longest_list
    childL=D[pattp]
    print('best parent=%s childL:'%pattp,childL)
    return pattp,childL

#...!...!....................
def deplete_hw_patterns(cntD,md,cntU):    
    minCs=md['readErrMit']['min_clique_size']
    nAtom=md['analyzis']['num_atom']
    print('DHW: inp size:%d nAtom=%d'%(len(cntD),nAtom))
    if len(cntD)<minCs: return None

    pattS=set(cntD)  # for new clique seed
    pattC=set(pattS) # for extension of clique
    print(pattS)
    outD={}

    while len(pattS) >=minCs:
        patt0=np.random.choice(tuple(pattS))        
        A=BitArray(hex=patt0)
        cliqueS={patt0} # begin of new clique
        for patt in pattC:
            B=BitArray(hex=patt)    
            X=A^B
            hw=X.count('1')  # Hamming weight
            if hw==2: cliqueS.add(patt)

        nClique=len(cliqueS)
        print('CLi %d  has %d candidates:'%(len(outD),nClique),cliqueS)

        if nClique<minCs:
            pattS.remove(patt0)
            continue

        pattp,childL=common_parent(cliqueS,nAtom)
        nChild=len(childL)
        if nChild<minCs:
            pattS.remove(patt0)
            continue
        
        print('accepted partent:',pattp)

        D={ patt:cntD[patt] for patt in childL }
        print('children:',D)
        ordCnt = sorted(D.items(), key=lambda x: x[1])
        mid=nChild//2
        print('ord:',ordCnt, 'nChild=%d mid=%d'%(nChild,mid))

        moveShot=0
        for i in range(mid):
            patt1,mshot1=ordCnt[i]
            patt2,mshot2=ordCnt[nChild-i-1]
            print('pp',patt1,patt2,'shots:',mshot1,mshot2)
            assert mshot2>=mshot1 , 'sorting failed'
            moveShot+=2*mshot1
            pattC.remove(patt1) # all counts were used
            pattS.remove(patt1) # match
            cntD[patt1]=0
            cntD[patt2]-=mshot1

        if 2*mid+1==nChild:  # odd size
            patt1,mshot1=ordCnt[mid]
            moveShot+=mshot1
            pattC.remove(patt1)
            pattS.remove(patt1) # match
            cntD[patt1]=0
        print('move %d shots to patt=%s'%(moveShot,pattp))
        if pattp not in cntU : cntU[pattp]=0
        cntU[pattp]+=moveShot
        

 

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=4)
    # Set a different seed based on the current time
    if args.rnd_seed!=0:
        np.random.seed(args.rnd_seed)
    else:
        np.random.seed(int(time.time()))
    

    amd={}
    amd['num_atom']=args.nAtom
    emd={'min_clique_size':3, 'min_hamm_weight':2}
    
    ftask=EmptyClass()
    ftask.verb=args.verb
    ftask.meta={'analyzis':amd,'readErrMit':emd}
    
    #... prepare true input
    cntD={}
    nshot=1000
    for bpatt in args.data_bin:
        patt=BitArray(bin=bpatt).hex
        cntD[patt]=nshot
        nshot+=500
    ftask.dataTrue=dict_2_expD(cntD)   

    cntD=generateReadError(ftask)
    ftask.dataNoisy=dict_2_expD(cntD)
    inpDD=split_by_Hamming_weight(ftask.dataNoisy)
    print('\nM: input noisy  DD:'); pprint(inpDD)
    
    cntD=mitigateReadErr(inpDD,ftask)
    
    
    print('M:done')


