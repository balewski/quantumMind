#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Postporocess   Problem Maximal-Independent-Set  experiment
reades meas/NAME.h5 as input
saves ana/NAME.mis.h5
Graphics optional
'''

import os
from pprint import pprint
import numpy as np
import json 

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from ProblemZ2Phase import ProblemZ2Phase
from PlotterZ2Phase import Plotter

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3,4],  help="increase output verbosity", default=1, dest='verb')
         
    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or env--> $QuEra_dataVault")

    parser.add_argument('-e',"--expName",  default='exp_62a15b',help='AWS-QuEra experiment name assigned during submission')
    parser.add_argument("--useHalfShots",  default=None,type=int, help="(optional) use 1st or 2nd half of the per-shot data, 0 uses all")

    # plotting
    parser.add_argument("-p", "--showPlots",  default='r s d', nargs='+',help="abc-string listing shown plots")
    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument( "-R","--mitReadErr",  action='store_true', default=False, help="enable readout error mitigation")

    args = parser.parse_args()
    # make arguments  more flexible
    if 'env'==args.basePath: args.basePath= os.environ['QuEra_dataVault']
    args.dataPath=os.path.join(args.basePath,'meas')
    args.outPath=os.path.join(args.basePath,'post')
    args.showPlots=''.join(args.showPlots)
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.outPath)
    return args


           

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=4)
                    
    inpF=args.expName+'.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))

    if 0: # back-hack 
        #from decimal import Decimal
        #expMD['payload']['evol_time_ramp_us']=Decimal('0.1')
        
        expMD['payload']['num_clust']=2
        expMD['payload']['num_atom_in_clust']=5
        expD['hamiltonian.JSON']=expD.pop('program_org.JSON')

         
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        print('M:bigD:',sorted(expD))
        rawBitstr=json.loads(expD['counts_raw.JSON'][0])
        rawShots=json.loads(expD['shots_raw.JSON'][0])
        print('M:bigD[shots_raw]:',len(rawShots))  #  before heralding
        print('M:bigD[counts_raw]:',len(rawBitstr)) # only valid, after heralding
        
        if args.verb>=3:
            for x in  expD:
                if 'qasm3' in x : continue
                print(x,expD[x])
  
        stop3

        
    task= ProblemZ2Phase(args,expMD,expD)

    task.postprocRawExperiment()
    if args.mitReadErr:
        from toolbox.MitReadErrQuEra import  MitigateReadErrorQuEra
        ana=MitigateReadErrorQuEra(task)
        #ana.insertReadError()
        #ana.mitigate()
        ok34_check_me

    task.energySpectrum()
    task.twoPointCorrelation()
    task.fit2ptCorr()
    
    
    #...... WRITE  OUTPUT
    outF=os.path.join(args.outPath,expMD['short_name']+'.z2ph.h5')
    write4_data_hdf5(task.expD,outF,expMD,verb=1)

    # ----  just plotting

    plot=Plotter(args,expMD['short_name'])

    #expMD['timeLR']=[0.,4.1]  # (ns)  time range clip
    if expMD['payload']['num_clust']==17:
        #expMD['XrangeLR']=[-5.,25.]  # (um)  x-range range clip
        expMD['YrangeLR']=[-5.,15.]  # (um)  x-range range clip
    
    if 'r' in args.showPlots:
        plot.show_register(task, what_to_draw="circle",figId=1)

    if 'd' in args.showPlots:  # drives
        if len(expMD['payload']['scar_time_us']):
            expMD['TrangeLR']=[0.,4.]  # (um)  x-range range clip
        plot.global_drive(task,figId=2)       
         
    if 's' in args.showPlots: # solutions
        nSol=min(6,expD['ranked_hexpatt'].shape[0])
        plot.Z2Phase_solutions( task,nSol=nSol ,figId=3)

    if 'c' in args.showPlots:  # 2D density  + two-point correlations
        plot.correlations(task,'density',figId=10)       

    if 'm' in args.showPlots:  
        plot.correlations(task,'magnet',figId=11)
        
    if 'e' in args.showPlots:  # energy spectrum
        #expMD['eneRangeLR']=[-3.4e-14,-2.2e-14]  # (um)  x-range range clip
        plot.energy_spectrum(task,figId=12)       

    if 'f' in args.showPlots:
        #expMD['maxNumOccur']=6;  expMD['numStateRange']=[0.7,500]
        plot.pattern_frequency(task,figId=13)       

    plot.display_all(png=1)

    print('M:done')


