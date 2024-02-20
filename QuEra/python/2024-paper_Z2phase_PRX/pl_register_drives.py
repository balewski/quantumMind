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
    def create_multi_canvas_layout(self,figId=1):
        """
        Create a multi-canvas layout with specified dimensions and axes.
        The layout includes one plot divided into 2 columns and the right column
        divided into 3 rows. Axes are assigned as specified and the right three axes share the x-axis.

        Returns:
        tuple: A tuple containing the figure and axes (axl, axr1, axr2, axr3).
        """
        figId=self.smart_append(figId)
        kwargs={'num':figId,'facecolor':'white', 'figsize':(10, 5)}
    
        # Create a figure with specified dimensions
        fig = self.plt.figure(**kwargs)

        # Define the grid layout
        gs = GridSpec(3, 2, figure=fig)

        # Create axes
        axl = fig.add_subplot(gs[:, 0])  # Left column
        axr1 = fig.add_subplot(gs[0, 1])  # First row of right column
        axr2 = fig.add_subplot(gs[1, 1], sharex=axr1)  # Second row of right column
        axr3 = fig.add_subplot(gs[2, 1], sharex=axr1)  # Third row of right column

        # Hide x-axis labels for axr1 and axr2 to avoid label overlapping
        self.plt.setp(axr1.get_xticklabels(), visible=False)
        self.plt.setp(axr2.get_xticklabels(), visible=False)

        return fig, (axl, axr1, axr2, axr3)

        

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    
    inpPath='/dataVault/dataQuEra_2023paper_confA/post'
    outPath='out/'

    expD1,expMD1=read4_data_hdf5(os.path.join(inpPath,'zurek_qpu_At0.5o58a.z2ph.h5'))
    expD2,expMD2=read4_data_hdf5(os.path.join(inpPath,'zurek_qpu_At1.5o58a.z2ph.h5'))
    expD3,expMD3=read4_data_hdf5(os.path.join(inpPath,'zurek_qpu_At3.0o58a.z2ph.h5'))

    #pprint(expMD)
    args=Stump()
    task=Stump()

    args.prjName='aquilaSetup'
    args.noXterm=True
    args.verb=1
    args.formatVenue='paper'
    args.outPath=outPath
    args.useHalfShots =None

    task1= ProblemZ2Phase(args,expMD1,expD1)
    task2= ProblemZ2Phase(args,expMD2,expD2)
    task3= ProblemZ2Phase(args,expMD3,expD3)
    
    
    # ----  just plotting
    plot=Plotter(args)
    fig, (axl, axr1, axr2, axr3)=plot.create_multi_canvas_layout()

    plot.show_register(task1, what_to_draw="circle",ax=axl)
    
    plot.global_drive(task1,axL=[axr1,None])
    plot.global_drive(task2,axL=[axr2,None])
    plot.global_drive(task3,axL=[axr3,None])    
    axr3.set_xlabel(r'Time ($\mu$s)')
   
    # ... add abcd
    axl.text(0.03,0.99,'a)',color='k', transform=axl.transAxes)
    axr1.text(0.03,0.92,'b)',color='k', transform=axr1.transAxes)
    axr2.text(0.03,0.92,'c)',color='k', transform=axr2.transAxes)
    axr3.text(0.03,0.92,'d)',color='k', transform=axr3.transAxes)

    plot.display_all(png=0)
    print('M:done')
