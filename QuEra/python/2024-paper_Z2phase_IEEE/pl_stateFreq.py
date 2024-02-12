#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# figure 1  for  Z2-phase  Quantum Computing and Engineering (QCE24)  paper 

import os, sys
sys.path.append('../problem_Z2Phase1D')

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.PlotterQuEra import PlotterQuEra
from ProblemZ2Phase  import ProblemZ2Phase 
from pprint import pprint
import numpy as np

class Stump:
    pass

from matplotlib.gridspec import GridSpec

#............................
#............................
#............................
class Plotter(PlotterQuEra):
    def __init__(self, args):
        PlotterQuEra.__init__(self,args)
#...!...!....................
    def create_canvas_row(self,figId=3):
        
        coreL=['zurek_qpu_At0.5o58aM4','zurek_qpu_At1.5o58aM4','zurek_qpu_At3.0o58aM4']        
        # ,'zurek_qpu_At2.1o58aM4'
        
        nrow,ncol=1,len(coreL)
        figId=self.smart_append(figId)
        kwargs={'num':figId,'facecolor':'white', 'figsize':(10, 3)}
        
        # Create the figure and subplots
        fig, axs = self.plt.subplots(nrow,ncol, sharey=True,**kwargs)
        
        for ic in range(ncol):
            ax=axs[ic]
            coreN=coreL[ic]
            inpF=os.path.join(inpPath,coreN+'.z2ph.h5')
            expD,expMD=read4_data_hdf5(inpF)
            task= ProblemZ2Phase(args,expMD,expD)
            expMD['maxNumOccur']=20;  expMD['numStateRange']=[0.7,2200]
            self.pattern_frequency(task,ax=ax)
            dLab=chr(97 + ic) + ')'
            ax.text(0.03,0.99,dLab,color='k', transform=ax.transAxes)
            if ic==0: ax.set( ylabel='Num. of states')
        return
        

        

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    
    inpPath='/dataVault/dataQuEra_2023paper_confA/post'
    outPath='out/'

    #pprint(expMD)
    args=Stump()
    
    args.prjName='stateFreq'
    args.noXterm=True
    args.verb=1
    args.formatVenue='paper'
    args.outPath=outPath
    args.useHalfShots =None
    
    # ----  just plotting
    plot=Plotter(args)
    plot.create_canvas_row()
    plot.display_all(png=0)
    print('M:done')
  
    
