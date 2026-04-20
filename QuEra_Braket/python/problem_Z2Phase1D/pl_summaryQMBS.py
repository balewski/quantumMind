#!/usr/bin/env python3
""" 
 concatenate mutiple measurements for
quantum many-body scar (QMBS) in a 1D chain.

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import  time
import sys,os
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.Util_readErrMit import mitigate_probs_readErr_QuEra_2bits
from toolbox.PlotterQuEra import PlotterQuEra

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("-o", "--outPath", default='out/',help="output path for plots and tables")

    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-p", "--showPlots",  default='ab', nargs='+',help="abc-string listing shown plots")


    args = parser.parse_args()
    args.showPlots=''.join(args.showPlots)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#............................
#............................
#............................
class Plotter(PlotterQuEra):
    def __init__(self, args):
        PlotterQuEra.__init__(self,args)

    #...!...!....................
    def scar_evolution(self,md,bigD,figId=1,ax=None,tit=None):
        if ax==None:
            nrow,ncol=1,1       
            figId=self.smart_append(figId)
            fig=self.plt.figure(figId,facecolor='white', figsize=(5.5,6))        
            # Create subplots within the specified figure
            ax, ax2 = fig.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

        timeV=bigD["evolTime_us"]
        dens2D=bigD["dens_evol"]
        nAtom=md['analyzis']['num_atom']

        if tit==None:
            tit='quantum many-body scar evolution, job=%s '%(md['short_name'])
        # Plot the 2D histogram, YlOrBr, , cmap='ocean'
        image = ax.imshow(dens2D, origin='lower', vmin=0, vmax=1)
        # Add a colorbar
        self.plt.colorbar(image, ax=ax, label='Rydberg density')
        ax.set(xlabel="time (us), after quench",ylabel="atom index")

        ax.axhline(nAtom-1.5); ax.text(0.15,0.95,'ancilla',transform=ax.transAxes,color='m')
        # User-defined tick lists
        nTime=timeV.shape[0]
        binsX = np.linspace(0,nTime-1,nTime)
        xLabs=[]

        timeOff=md['payload']['scar_time_start_us']
        print('tt',timeOff)
        for i in range(0,nTime):
            if i%3!=0: xLabs.append('')
            else: xLabs.append('%.1f'%(timeV[i] + timeOff))

        print('shT:',xLabs, '\nbinsX',binsX)
        print('shd',dens2D.shape)
        
        ax.set_xticks(binsX)
        ax.set_xticklabels(xLabs)

        #.... add driving field  in the main(.)
        
        return ax2
  
        #.... mis text
        txt1='device: '+md['submit']['backend']
        txt1+='\nexec: %s'%md['job_qa']['exec_date']
        txt1+='\ndist %s um'%(md['payload']['atom_dist_um'])
        txt1+='\nOmaga %.1f MHz'%md['payload']['rabi_omega_MHz']
        txt1+='\nshots/job: %d'%md['submit']['num_shots']
        txt1+='\ntot atoms: %d'%md['payload']['tot_num_atom']
        txt1+='\nnum jobs: %d'%nTime
        txt1+='\nreadErr eps: %s'%md['analyzis']['readErr_eps']
        ax.text(0.02,0.55,txt1,transform=ax.transAxes,color='g')

        return
    
   
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
    #print('sub-dirs',jobL)

    sumL=[]
    for sumF in jobL:
        if '.h5' not in sumF: continue
        if pattern not in sumF: continue
        sumL.append(sumF)
        
    if verb>0: print('found %d sum-files'%len(sumL),sumL[:3],'...')
    return sorted(sumL)

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()
    coreN='scar13atom'    
    device='cpu'
    
    anaPath='/dataVault/dataQuEra_2023julyQMBS_%s/ana'%device  
    sumFL=locateSummaryData( anaPath,pattern=coreN+'_'+device)
    assert len(sumFL) >0
    #sumFL=sumFL[:5] #tmp, for testing
      
    evolTime=[]
    densR=[]  # dens_rydberg (12,) 
    for i,inpF in enumerate(sumFL):        
        expD,expMD=read4_data_hdf5(os.path.join(anaPath,inpF),verb=i==0)
        if i==0: pprint(expMD)
        tscarL=expMD['payload'].pop('scare_time_us')
        tEvol=float(tscarL[1])
        evolTime.append(tEvol)
        densR.append(expD['dens_rydberg'])

    # convert to numpy
    densR=np.array(densR) # shape[  nAtom, nTime,]
    expMD['short_name']= '%s.m%d.%s'%(coreN,len(sumFL),device)
    t0=expMD['payload']['evol_time_us']
    t1,t2=tscarL
    expMD['payload']['scar_time_start_us']=t0+2*t1
    bigD={}
    bigD["evolTime_us"]=np.array(evolTime)
    bigD["dens_evol"]=np.swapaxes(densR,0,1)  # [nTime, nAtom]

    print('M: evolTime/us:',evolTime, bigD["dens_evol"].shape)
        
    # ----  just plotting
    args.prjName=expMD['short_name']
    plot=Plotter(args)
    if 'a' in args.showPlots:
        ax2=plot.scar_evolution(expMD,bigD,figId=1)
        from ProblemZ2Phase import ProblemZ2Phase
        task= ProblemZ2Phase(args,expMD,expD)
        expMD['TrangeLR']=[0.,4.]  # (um)  x-range range clip
        plot.global_drive(task,axL=[ax2,None])       

    plot.display_all(png=1)
    

