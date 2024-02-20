#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Generate ideal (fake) Z2 data with controlled amount of wall density number 
- no graphics

'''

import sys,os
import time
from pprint import pprint
import numpy as np

import sys
sys.path.append('../problem_Z2Phase1D')

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.UAwsQuEra_job import submit_args_parser, build_ranked_hexpatt_array
from toolbox.UAwsQuEra_job import  harvest_retrievInfo, postprocess_job_results 
from ProblemZ2Phase import ProblemZ2Phase
from Util_zurek import accumulate_wallDens_histo
from collections import Counter

import argparse


#...!...!..................
def get_parser(backName="cpu"):  # add task-speciffic args
    parser = argparse.ArgumentParser()

    parser.add_argument('--chain_shape', default=['loop',58],  nargs='+', help=' line [n] OR  OR uturn [n] OR  loop [n],  space separated list')

    parser.add_argument('--bitInfidelity', default=0.05,  type=float, help=' probability of flipping any bit OR -k flip exctly k  random  bits ')
    
    # ....  do not use those - any value is fine .....
    
    #  ProblemMIS task speciffic
    parser.add_argument('--atom_dist_um', default='6.2', type=str, help='distance between 2 atoms, in um')

    parser.add_argument('--detune_shape', default=[0.25,0.5,0.75], type=float, nargs='+', help='relative intermediate values of detune, space separated list')
    parser.add_argument('--scar_time_us', default=[], type=float, nargs='+', help='free evolution to produce QMBS, [ ramps_us, flat_us]; [] is OFF, space separated list')
    parser.add_argument( "-M","--multi_clust",   action='store_true', default=False, help="replicate  cluster multipl times")
    parser.add_argument('--hold_time_us', default=0, type=float,help='delay measurement, hold last value of Delta and Omega')

    
    args=submit_args_parser(backName,parser)
    assert len(args.scar_time_us) in [0, 2]
    assert args.bitInfidelity <=1.0
    args.outPath=args.outPath.replace('jobs','post')
    return args

#...!...!..................
def pack_4_write( bitpattL,wallMoments,expD,md,md1):
    pmd= md['payload']
    ppmd=md['postproc']
    nAtom=pmd['num_atom_in_clust']
    
    solCounter=Counter(bitpattL) #.... convert list to dict
    nSol=len(solCounter)
    uShots=len(bitpattL)
    print(' nSol=%d  shots used=%d  '%(nSol,uShots))

    #...... collect meta-data
    #pprint(ppmd)    
    ppmd['num_atom']=nAtom
    ppmd['num_sol']=nSol
    ppmd['num_fail']=0
    ppmd['used_shots']=uShots

    print('wm', wallMoments)
    fmd={}
    fmd['name']='Z2 phase, domain wall density'
    fmd['mean_wall_number']=[wallMoments[0],wallMoments[1]]
    fmd['std_wall_number']=[wallMoments[2],wallMoments[3]]
    fmd.update(md1)
    pmd['ideal_sample']=fmd
    
    # ... final bigD packing
    #print('expD:',expD.keys())

    countsV,hexpattV,hammV,nAtom2=build_ranked_hexpatt_array(solCounter)
    assert nAtom==nAtom2
        
    expD['ranked_counts_ideal']= countsV   # [NB]
    expD['ranked_hexpatt_ideal']=hexpattV  # [NB]
    
     
#...!...!..................
def XXXwall_pattern_A_dist(z2patt):
    #  always flip 2  randomly bits which are at least 4 positions apart
 
    outL=[ 'x'  for i in range(args.numShots) ]
    for j in range(args.numShots):
        dist=0
        while dist<5:
            idxL = np.random.choice(np.arange(nAtom), size=2, replace=False)
            dist1=abs(idxL[0] -idxL[1])
            dist2=abs(idxL[0] -idxL[1]+nAtom)  # wrupa-up case
            dist=min(dist1,dist2)
        #print(idxL,dist)
        patt=[int(i) for i in z2patt]
        for ix in idxL:  patt[ix]=1-patt[ix]
        pattS=''.join([str(x) for x in patt])
        #print('patt',j,pattS)
        outL[j]=pattS

    # .... verify wall density
    wallDensV,wallMoments=accumulate_wallDens_histo( outL,isLoop=nAtom==58)
    return outL,wallMoments

#...!...!..................
def wall_pattern_A(z2patt, nFlip):
    print('always flip %d bits, randomly'%(nFlip))
    outL=[ 'x'  for i in range(args.numShots) ]
    for j in range(args.numShots):
        idxL = np.random.choice(np.arange(nAtom), size=nFlip, replace=False)
        patt=[int(i) for i in z2patt]
        for ix in idxL:  patt[ix]=1-patt[ix]
        pattS=''.join([str(x) for x in patt])
        outL[j]=pattS

    # .... verify wall density
    wallDensV,wallMoments=accumulate_wallDens_histo( outL,isLoop=nAtom==58)
    return outL,wallMoments

#...!...!..................
def wall_pattern_B(z2patt, pFlip=0.1):
    #  randomly flip any bit with prob pflip
    print('flip all bits with prob %.3f'%(pFlip))
    outL=[ 'x'  for i in range(args.numShots) ]
    for j in range(args.numShots):
        randP=np.random.rand(nAtom)
        patt=[int(i) for i in z2patt]
        for i in range(nAtom):
            if randP[i] <pFlip :  patt[i]=1-patt[i]
        
        pattS=''.join([str(x) for x in patt])
        #print('patt',j,pattS)
        outL[j]=pattS

    # .... verify wall density
    wallDensV,wallMoments=accumulate_wallDens_histo( outL,isLoop=nAtom==58)
    return outL,wallMoments
    

    
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    task=ProblemZ2Phase(args)     
    task.placeAtoms()
    task.buildHamiltonian()
    
    ahs_program=task.buildProgram()  # SchrodingerProblem   
    jobMD=task.submitMeta

    task.postprocess_submit(None)
    task.submitMeta['short_name']=task.submitMeta['short_name']
    
    if args.verb>1 : pprint(jobMD)

    #.... prepare a perfect Z2phase pattern
    nAtom= jobMD['payload']['num_atom_in_clust']
    z2patt=['0']*nAtom
    for i in range(nAtom//2): z2patt[i*2]='1'
    print('z2pat(%d)=%s'%(nAtom,''.join(z2patt)))
    
    if args.bitInfidelity<0:
        nfb=int(-args.bitInfidelity)
        bitpattL,wallMoments=wall_pattern_A(z2patt, nfb)
        md1={'num_flipped_bits' : nfb}        
    else:
        bitpattL,wallMoments=wall_pattern_B(z2patt, pFlip=args.bitInfidelity)
        md1={'bit_infidelity' : args.bitInfidelity}
    pack_4_write( bitpattL,wallMoments,task.expD,jobMD,md1)
        
    
    jobMD=task.submitMeta
    
    if args.verb>0: pprint(jobMD)
    #...... WRITE  JOB META-DATA .........
    outF=os.path.join(args.outPath,jobMD['short_name']+'.z2ph.h5')
    write4_data_hdf5(task.expD,outF,jobMD)
    print('M:end --expName   %s   %s  %s  ARN=%s'%(jobMD['short_name'],jobMD['hash'],args.backendName ,jobMD['submit']['task_arn']))
    

    baseStr= " --basePath %s "%args.basePath if args.basePath!=os.environ['QuEra_dataVault']  else ""  # for next step program
    print('   ./apply_zneWall.py     %s --expName   %s -X  --noiseScale  1.2 1.4 1.6 1.8 2.0  \n'%(baseStr,jobMD['short_name'] ))
    
    
