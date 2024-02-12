#!/usr/bin/env python3
""" 
 Kibble-Zurek scaling analysis

Apply Zero Noise Extrapolation to wall density  mean and std

Input : single measurement
     zurek_qpu_td3.0o58a.z2ph.h5 
Output: raw and corrected wall density  & its moments
     zurek_qpu_td3.0o58a.zneWall.h5
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import  time
import sys,os
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from Sampler4ReadErrZNE import Sampler4ReadErrZNE
from FitterReadErrZNE import FitterReadErrZNE
from toolbox.ModelLinear import ModelLinear
from toolbox.ModelQuadratic import ModelQuadratic


import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or env--> $QuEra_dataVault")

    parser.add_argument('-e',"--expName",  default='zurek_qpu_td3.0o58a',help='[.z2ph.h5] name of analyzed  experiment ')
    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-p", "--showPlots",  default='df', nargs='+',help="abc-string listing shown plots")
    parser.add_argument( "--readoutError",  default=[0.01, 0.08] , type=float, nargs='+', help="readout error for [ set0meas1, set1meas0 ]")
    parser.add_argument('--noiseScale',type=float,default=[ 1.2, 1.4, 1.6, 1.8, 2.0 ], nargs='+', help="noise ampliffication, list space separated")
    parser.add_argument("--sampleReuse",  default=40, type=int, help='sample reuse factor')

    args = parser.parse_args()
    # make arguments  more flexible
    if 'env'==args.basePath: args.basePath= os.environ['QuEra_dataVault']
    #args.basePath='/dataVault/dataQuEra_2023paper_take1/'  #tmp
    args.dataPath=os.path.join(args.basePath,'post')
    args.outPath=os.path.join(args.basePath,'rezne')
    args.showPlots=''.join(args.showPlots)
    args.onlySPAM=True  # for Milo to not run ZNE, tmp
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.outPath)
    assert len(args.readoutError)==2
    

    return args

    
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()
    np.set_printoptions(precision=4)
                    
    inpF=args.expName+'.z2ph.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))
         
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        stop3

    sampler=Sampler4ReadErrZNE(expD,expMD,args)
    if  args.onlySPAM:
        sampler.finalize()
        outF=os.path.join(args.outPath,expMD['short_name']+'.zneWall.h5')
        write4_data_hdf5(expD,outF,expMD)
        exit(0)

    ok99
    sampler.amplify_SPAM()
    sampler.finalize()


    isIdealSample='ideal_sample' in expMD['payload']
    fitZNE=FitterReadErrZNE(args)
    
    #oscModel=ModelLinear()
    oscModel=ModelQuadratic()
    momentV=np.swapaxes(expD["domain_wall_dens_moments"],0,1)  # [PEY,nTime]
    noiseV=expD["SPAM_scale"]
    print('\nM: prep to fit ZNE')
    print('M:momentV',momentV)
    print('M:noiseV',noiseV)

    
    probData=momentV[:3,:]    # fit input mean X 
    #probData=momentV[2:5,:]  # fit input std X 
    
    print('M:probData',probData.shape)

    if isIdealSample :  # skip ground truth
        probData=probData[:,1:]
        noiseV=noiseV[1:]

    #1probData[1]*=2. # testing doubling of error
    fitZNE.fit_model(probData,noiseV, oscModel )

    # summary line
    extrX=fitZNE.data['extraX']
    extrY=fitZNE.data['extraY']
    extrE=fitZNE.data['extraYerr']
    fqa=fitZNE.fitMD['fit_result']['fitQA']
    nfree=fqa['nfree'][0]
    redchi=fqa['redchi'][0]
    
    print('\nM: measured: mean X=%.2f +/-%.2f, std X=%.2f ; REZNE mean X=%.2f +/-%.2f  nfree=%d  chi2/dof=%.2g '%(probData[0,0],probData[1,0],probData[2,0],extrY[0],extrE[0],nfree,redchi))
    print('M:goldZNE,%.3f,%.3f,%.3f,,%.3f,%.3f,%d,%.2g\n'%(probData[0,0],probData[1,0],probData[2,0],extrY[0],extrE[0],nfree,redchi))

    
    #...... WRITE  JOB META-DATA .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.zneWall.h5')
    write4_data_hdf5(expD,outF,expMD)
    

    # ----  just plotting
    if 'd' in args.showPlots:
        expMD['densLR']=[0.,40.]  
        fitZNE.draw_domain_wall_density(expMD,expD,figId=1)
    if 'f' in args.showPlots:
        fitZNE.draw_ZNEfit(expMD,expD,figId=2)
    fitZNE.display_all(png=1)
   
