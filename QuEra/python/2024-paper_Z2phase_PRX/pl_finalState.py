#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# figure 2  for  Z2-phase  Quantum Computing and Engineering (QCE24)  paper 

import os, sys
sys.path.append('../problem_Z2Phase1D')

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
#from toolbox.PlotterQuEra import PlotterQuEra
from pprint import pprint
from ProblemZ2Phase  import ProblemZ2Phase 
from PlotterZ2Phase import Plotter

class Stump:
    pass


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    
    inpPath='/dataVault/dataQuEra_2023paper_confA/post'
    outPath='out/'

    expD,expMD=read4_data_hdf5(os.path.join(inpPath,'zurek_qpu_At3.0o58b.z2ph.h5'))

    #pprint(expMD)
    args=Stump()
    task=Stump()

    args.prjName='finalState'
    args.noXterm=True
    args.verb=1
    args.formatVenue='paper'
    args.outPath=outPath
    args.useHalfShots =None
 
    task= ProblemZ2Phase(args,expMD,expD)
    
    # ----  just plotting
    plot=Plotter(args)

    nSol=3
    ax=plot.Z2Phase_solutions( task,nSol=nSol ,figId=2)

    # Adding text 'abc' at (4, 1)
    ax.text(20, 15, 'defect',size=20)

    # Adding an arrow from the text to the point (-1, 1)
    ax.annotate('', xy=(12, 6), xytext=(20, 15),
                arrowprops=dict(facecolor='black', arrowstyle="->"))

    plot.display_all(png=0)

   
    print('M:done')
