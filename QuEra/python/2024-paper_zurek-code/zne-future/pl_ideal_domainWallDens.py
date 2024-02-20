#!/usr/bin/env python3
""" 
agregated comparison of ideal and SPAM affected domain wall density
ideal simulations
paper-quality 
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import  time
import sys,os
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.Util_readErrMit import mitigate_probs_readErr_QuEra_2bits
from toolbox.PlotterBackbone import PlotterBackbone

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("-o", "--outPath", default='out/',help="output path for plots and tables")

    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-p", "--showPlots",  default='d', nargs='+',help="abc-string listing shown plots")


    args = parser.parse_args()
    args.showPlots=''.join(args.showPlots)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)
#...!...!....................
    def agregated_domain_wall_density(self,md,dens3D,moment2D,figId=1):
        
        pmd=md['payload']
        zmd=md['zne_wall_dens']
        
        isIdealSample='ideal_sample' in pmd
        assert isIdealSample
        
        nAtom= md['postproc']['num_atom'] 
        K=dens3D.shape[0]
        assert dens3D.shape[1]>=2
        densX = np.arange(dens3D.shape[2])
            
        tLab='Ideal+SPAM simu of Z2-phase (%d-atoms)'%(nAtom)       
        spamTxt='SPAM s0m1: %.2f   s1m0: %.2f'%(zmd['readout_error'][0],zmd['readout_error'][1])
        print('spamTxt:',spamTxt)
        
        nrow,ncol=K,1       
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,10))
        axs= fig.subplots(nrow,ncol,sharex=True)

        axs[0].set_title(tLab)
        yy=0.05
        
        
        for k in range(K):
            ax=axs[k]

            # Create a bar plot to visualize the histogram
            dCol='gold'; dLab='truth'
            ax.bar(densX,dens3D[k][0] , width=1., edgecolor='black', color=dCol,label=dLab)
            avrX,avrXer,stdX,stdstd,mshots=moment2D[k][0]
            ax.plot(avrX, yy ,'D',color='r',markersize=8, label='truth avr')

            
            dCol='lime'; dLab='+SPAM'
            ax.bar(densX,dens3D[k][1] , width=1., edgecolor='black', color=dCol,label=dLab, alpha=0.5)
            avrX,avrXer,stdX,stdstd,mshots=moment2D[k][1]
            ax.plot(avrX, yy ,'o',color='b',markersize=8,label='+SPAM avr')
            
            ax.set_ylabel('probability')
                            
            if k==K-1: # bottom row
                ax.set(xlabel="wall density number")           

            if k==0:
                ax.text(0.4,0.2,spamTxt,transform=ax.transAxes,color='b')
           
        axs[0].legend()
        if 'densLR' in md:   axs[0].set_xlim(tuple(md['densLR']))



#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()
   
    dataPath='/dataVault/dataQuEra_2023julyA/rezne/'

    pfbitL=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    
    
    sumFL=[   'ideal_o58pf%.2f.zneWall.h5'%x for x in  pfbitL ]
    print('sumFL:',sumFL)
    dens3D=[]
    moment2D=[]
    
    for i,inpF in enumerate(sumFL):        
        expD,expMD=read4_data_hdf5(os.path.join(dataPath,inpF),verb=i==0)
        if i==0: pprint(expMD)
        isIdealSample='ideal_sample' in  expMD['payload']
        #nAtom= md['postproc']['num_atom'] 

        dens2D=expD["domain_wall_dens_histo"]
        noiseV=expD["SPAM_scale"]
        momentV=expD["domain_wall_dens_moments"]
        
        dens3D.append(dens2D)
        moment2D.append(momentV)
                                               
    # convert to numpy
    dens3D=np.array(dens3D) # shape[  nAtom, nTime,]
    print('dens3D:',dens3D.shape)

    # ----  just plotting
    args.prjName=expMD['short_name']
    plot=Plotter(args)
    if 'd' in args.showPlots:
        expMD['densLR']=[-1.,35.]
        plot.agregated_domain_wall_density( expMD,dens3D,moment2D)

    plot.display_all(png=1)
 
