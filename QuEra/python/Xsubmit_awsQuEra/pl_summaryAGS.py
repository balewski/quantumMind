#!/usr/bin/env python3
""" 
 concatenate mutiple measurements from Quera or from emulation
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import  time
import sys,os
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.PlotterBackbone import PlotterBackbone
from toolbox.Util_stats import do_yield_stats  # for heralding
from toolbox.Util_readErrMit import mitigate_probs_readErr_QuEra_1bit

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("-o", "--outPath", default='out/',help="output path for plots and tables")

    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-p", "--showPlots",  default='ab', nargs='+',help="abc-string listing shown plots")
    parser.add_argument('--readErrEps', default=None, type=float, help='probability of state 1 to be measured as state 0')


    args = parser.parse_args()
    args.showPlots=''.join(args.showPlots)
    if 'h' in args.showPlots: assert len(args.showPlots)==1  # conflict of plotters

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

    #...!...!....................
    def rabiOscAll(self,md,bigD,figId=1,ax=None,tit=None):
        if ax==None:
            nrow,ncol=1,1       
            figId=self.smart_append(figId)
            fig=self.plt.figure(figId,facecolor='white', figsize=(6,4))
            ax = self.plt.subplot(nrow,ncol,1)


        timeV=bigD["evolTime"]
        
        probsAtom=bigD["probsAtom"]
        probsSum=bigD["probsSum"]
        print('ppA',probsAtom.shape,probsSum.shape)

        nAtom=probsAtom.shape[2]
        nTime=timeV.shape[0]

        if tit==None:
            tit='single atom evolution, job=%s  m:%d'%(md['short_name'],nTime)

        dMkL=['o','D','^','s','*','+','x','>','1']
            
        for k in range(nAtom):
            if k>6: break
            Y=probsAtom[0,:,k]
            erY=probsAtom[1,:,k]
            dLab='atom %d'%k
            dMk=dMkL[k]
            ax.errorbar(timeV, Y,yerr=erY,label=dLab,marker=dMk)
            #1print('atom %d probs:'%k,Y)
        
        Ys=probsSum[0,:,1]*nAtom
        erYs=probsSum[1,:,1]*nAtom

        print('sum yiled',Ys, erYs)
        
        ax.errorbar(timeV, Ys,yerr=erYs,label='prob sum',marker='o',lw=2.,ms=6)
        ax.axhline(nAtom,c='k',ls='--')
        
        lTit='dist %s um'%(md['payload']['atom_dist_um'])
        
        ax.legend(title=lTit, loc='upper right') # center upper 
        ax.set(xlabel='evolution time (us)',ylabel='r-state probability',title=tit)
        ax.grid()
        ax.set_ylim(-0.05,nAtom+0.5)
        #ax.set_xlim(2.75,3.45)

        #.... mis text
        txt1='device: '+md['submit']['backend']
        txt1+='\nexec: %s'%md['job_qa']['exec_date']
        txt1+='\ntot atoms: %d'%nAtom
        txt1+='\nshots/job: %d'%md['submit']['num_shots']
        txt1+='\nnum jobs: %d'%nTime
        txt1+='\nreadErr eps: %s'%md['analyzis']['readErr_eps']

        ax.text(0.02,0.55,txt1,transform=ax.transAxes,color='g')

        return
    
    #...!...!....................
    def rabiOscSub(self,md,bigD,byColumn=True, figId=1):
        figId=self.smart_append(figId)
        nrow,ncol=2,2
        nrow,ncol=4,1
        
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,16))

        probsAtom=bigD["probsAtom"] # shape[ PAY,nTime, atom]
        if byColumn:
            probsSum=bigD["probsCol"]  # shape[ PAY,nTime, col]
            tit0='column='
        else:
            probsSum=bigD["probsRow"]  # shape[ PAY,nTime, row]
            tit0=' row='
                     
        print('ppS',probsAtom.shape,probsSum.shape)
        nSub=4
        nSubAtom=probsAtom.shape[-1] // probsSum.shape[-1] 
        nTime=probsAtom.shape[1]
        for k in range(nSub):
            iat=k*nSubAtom
            D={"evolTime":bigD["evolTime"]} # is common
            if byColumn:
                D["probsAtom"]=probsAtom[...,k::nSubAtom]
            else:
                D["probsAtom"]=probsAtom[...,iat:iat+nSubAtom]
            D["probsSum"] =probsSum[...,k].reshape(-1,nTime,1)
            tit=tit0+'%d, 1atom evol, job=%s  m:%d'%(k,md['short_name'],nTime)
            ax = self.plt.subplot(nrow,ncol,1+k)
            self.rabiOscAll(md,D,ax=ax,tit=tit)

            #break

#...!...!....................
def locateSummaryData(src_dir,pattern=None,verb=1):

    for xx in [ src_dir]:
        if os.path.exists(xx): continue
        print('Aborting on start, missing  dir:',xx)
        exit(1)

    if verb>0: print('locate summary data src_dir:',src_dir,'pattern:',pattern)

    jobL=os.listdir(src_dir)
    print('locateSummaryData got %d potential jobs, e.g.:'%len(jobL), jobL[0])
    print('sub-dirs',jobL)

    sumL=[]
    for sumF in jobL:
        if '.h5' not in sumF: continue
        if pattern not in sumF: continue
        sumL.append(sumF)
        
    if verb>0: print('found %d sum-files'%len(sumL))
    return sorted(sumL)

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()
    anaPath='/dataVault/dataQuEra_2023mayI/ana'  # dist 25 um ,  for 1-atom Rabi, T<1.7 us
    anaPath='/dataVault/dataQuEra_2023mayJ/ana'  # dist 25 um ,  for 1-atom Rabi, 2.8<T<4 us

    device='qpu'; anaPath='/dataVault/dataQuEra_2023may1AtA_%s/ana'%device  # dist24 um, T=[0.3,1.7] um, 7 shots, 9 atoms
    #device='qpu';anaPath='/dataVault/dataQuEra_2023may1AtB_%s/ana'%device  # dist24 um, T=[2.8,4.0] um, 7 shots, 9 atoms
    #anaPath='/dataVault/dataQuEra_2023may1AtS_qpu/ana'  # dist24 um, T=[2.8,3.0] um, 7 shots, 9 atoms
    #device='qpu'; anaPath='/dataVault/dataQuEra_2023may1B_%s/ana'%device  # dist21 um,# readErrMit  1-to-0
    #device='qpu'; anaPath='/dataVault/dataQuEra_2023may1C_%s/ana'%device # dist21 um,# readErrMit  only 0s
   
    sumFL=locateSummaryData( anaPath,pattern='rabi_'+device)
    assert len(sumFL) >0

    evolTime=[]
    probsA=[]  # per atom
    probsS=[]  # sum over all atoms
    probsC=[]  # per column
    probsR=[]  # per row
    heraldY=None
    for i,inpF in enumerate(sumFL):        
        expD,expMD=read4_data_hdf5(os.path.join(anaPath,inpF),verb=i==0)
        if i==0: pprint(expMD)
        
        tEvol=float(expMD['payload'].pop('evol_time_us'))
        evolTime.append(tEvol)

        probsA.append(expD['prob_atom'])
        probsS.append( expD['prob_sum'][:,0,:] )# skip measurement index
        #1probsC.append(expD['prob_col'])
        #1probsR.append(expD['prob_row'])
        if i==0:   # shape[ atom, NB]
            heraldY=expD['herald_counts_atom'].copy()
        else:
            heraldY+=expD['herald_counts_atom'] 
    # convert to numpy
    probsA=np.array(probsA) # shape[nTime, PAY, atom, NB]
    probsS=np.array(probsS)
    probsC=np.array(probsC) # shape[nTime, PAY, col, NB]
    probsR=np.array(probsR)
 
    #print("a1",heraldY.shape)
    heraldP=do_yield_stats(heraldY)
    #print("a2",heraldP.shape)
    
    bigD={}
    bigD["evolTime_us"]=np.array(evolTime)
    bigD["probsAtom"]=np.swapaxes(probsA,0,1)[...,1]  # rydberg state probability
    bigD["probsSum"]=np.swapaxes(probsS,0,1)
    #1bigD["probsCol"]=np.swapaxes(probsC,0,1)[...,1]  # shape[ PAY,nTime, col]
    #1bigD["probsRow"]=np.swapaxes(probsR,0,1)[...,1]
    bigD["herald_prob_atom"]=heraldP  # special case
    print('M: evolTime/us:',evolTime, bigD["probsAtom"].shape, bigD["probsSum"].shape)
    #... apply readErr Mitigation to the sum
    eps=args.readErrEps
    expMD['analyzis']['readErr_eps']=eps
    if eps!=None:
        dataP=bigD["probsSum"]
        #print('\nsum Probs before   eps=%.2f\n'%eps,dataP.shape)
        dataM=mitigate_probs_readErr_QuEra_1bit(dataP,eps)
        #print('\nafter  \n',dataM)
        bigD['probsSum']=dataM

        
    # ----  just plotting
    args.prjName=expMD['short_name']+'m%d'%expMD['payload']['num_atom']
    plot=Plotter(args)
    if 'a' in args.showPlots:
        plot.rabiOscAll(expMD,bigD,figId=1)
    if 'c' in args.showPlots:
        plot.rabiOscSub(expMD,bigD,byColumn=True,figId=2)
    if 'r' in args.showPlots:
        plot.rabiOscSub(expMD,bigD,byColumn=False,figId=3)
    if 'h' in args.showPlots:  # special case
        from ana_AGS import Plotter as Plotter2
        plot2=Plotter2(args)
        plot2.heralding_per_atom(expMD,bigD,figId=4)
        plot2.display_all(png=1)
        exit(0)
    plot.display_all(png=1)
