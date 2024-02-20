#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Analyze   Problem Maximal-Independent-Set  experiment
reades meas/NAME.h5 as input
saves ana/NAME.mis.h5
Graphics optional
'''

import os
from pprint import pprint
import numpy as np
import json 
import networkx as nx  # for graph generation
from bitstring import BitArray

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from ProblemMaxIndependentSet import ProblemMIS
from toolbox.PlotterQuEra import PlotterQuEra

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3,4],  help="increase output verbosity", default=1, dest='verb')
         
    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or env--> $QuEra_dataVault")

    parser.add_argument('-e',"--expName",  default='exp_62a15b',help='AWS-QuEra experiment name assigned during submission')

    # plotting
    parser.add_argument("-p", "--showPlots",  default='r s d', nargs='+',help="abc-string listing shown plots")
    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False, help="disable X-term for batch mode")
    
    args = parser.parse_args()
    # make arguments  more flexible
    if 'env'==args.basePath: args.basePath= os.environ['QuEra_dataVault']
    args.dataPath=os.path.join(args.basePath,'meas')
    args.outPath=os.path.join(args.basePath,'ana')
    args.showPlots=''.join(args.showPlots)
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.outPath)
    return args


#............................
#............................
#............................
class Plotter(PlotterQuEra):
    def __init__(self, args,prjName=None):
        if prjName!=None: args.prjName=prjName
        PlotterQuEra.__init__(self,args)
   
      
    #...!...!....................
    def MIS_solutions(self,task,nSol,figId=3):
        figId=self.smart_append(figId)
        nrow,ncol=2,round(0.1+nSol/2)

        fig=self.plt.figure(figId,facecolor='white', figsize=(4*ncol,4.5*nrow))
        md=task.meta
        smd=md['submit']
        amd=md['analyzis']
        pd=md['payload']
        nAtom= amd['num_atom']
        
        G=task.graph
        
        pos=task.atoms_pos_um()[:nAtom]
        # Print the graph
        #print('nodes:',G.nodes); print(G.edges)
        #print('pos_um',pos)
        
        probV = task.expD['ranked_probs']
        hexpattV= task.expD['ranked_hexpatt']
        hammwV=task.expD['ranked_hammw']  
        energyV=task.expD['ranked_energy_eV']

        nSigThr=1.  # cuto-off for signifficance of results
        for k in range(nSol):
           ax = self.plt.subplot(nrow,ncol,1+k)
           ax.set_aspect(1.)
           
           hexpatt=hexpattV[k]
           A=BitArray(hex=hexpatt)[-nAtom:]  # clip leading 0s
           prob,probEr,mshot=probV[:,k]
           card=hammwV[k]
           nSig=prob/probEr
           print('patt=',hexpatt,prob,probEr,mshot,'nSig=%.1f'%nSig)
           #1if nSig<nSigThr: continue
           vert_colors = ['red' if (bitc == '1') else 'aqua' for bitc in A.bin]
           #print('cols:',vert_colors)
           
           tit='i:%d patt:%s  prob: %.3f+/-%.3f'%(k,hexpatt,prob,probEr)
           tit+='\ncard:%d   mshot:%d/%d\nene:%.2g (eV)'%(card,mshot,amd['used_shots'],energyV[k])
           #tit+='\ncard:%d   mshot:%d/%d'%(card,mshot,amd['used_shots'])
           #print('sol:',tit)
           ax.set(title=tit)
           #print('colL:',vert_colors)
           nx.draw(G, pos = pos, ax=ax, with_labels=True, node_color=vert_colors)

           txt1=None
           if k==0: txt1=md[ 'payload']['info'].replace(",","\n")
           if k==1: txt1=md['submit']['info'].replace(",","\n")
           if k==2: txt1=md['job_qa']['info'].replace(",","\n")
           if txt1!=None: ax.text(0.02,0.89,txt1,color='g', transform=ax.transAxes,va='top')
           

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
        if args.verb>=3:
            for x in  expD:
                if 'qasm3' in x : continue
                print(x,expD[x])
  
        stop3

        
    task= ProblemMIS(args,expMD,expD)

    task.analyzeRawExperiment()
    task.energySpectrum()
     
    #...... WRITE  OUTPUT
    outF=os.path.join(args.outPath,expMD['short_name']+'.mis.h5')
    write4_data_hdf5(task.expD,outF,expMD)

    # ----  just plotting

    plot=Plotter(args,expMD['short_name'])

    #expMD['timeLR']=[0.,4.1]  # (ns)  time range clip
    if expMD['payload']['num_clust']==1:
        expMD['XrangeLR']=[-5.,25.]  # (um)  x-range range clip
        expMD['YrangeLR']=[-5.,25.]  # (um)  x-range range clip
    
    if 'r' in args.showPlots:
        plot.show_register(task, what_to_draw="circle")
         
    if 'd' in args.showPlots:  # drives
        plot.global_drive(task)
        
    if 's' in args.showPlots: # solutions
        nSol=min(6,expD['ranked_hexpatt'].shape[0])
        plot.MIS_solutions( task,nSol=nSol )

    if 'e' in args.showPlots:  # energy spectrum
        #expMD['eneRangeLR']=[-3.4e-14,-2.2e-14]  # (um)  x-range range clip
        plot.energy_spectrum(task,figId=12)       

    plot.display_all(png=1)

    print('M:done')


