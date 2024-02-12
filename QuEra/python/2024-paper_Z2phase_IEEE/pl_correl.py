#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import matplotlib.pyplot as plt
import numpy as np
import os

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.PlotterBackbone import roys_fontset


def plot_image_with_top_colorbar(expD,md,ax,corType):
  
    amd=md['postproc']
    pd=md['payload']
    nAtom= amd['num_atom']

    if corType=='density':
       data2d=expD['dens_corr']
       vmax=0.25; vmin=-vmax
       dLab='a)'
       ax.set(ylabel="atom index")

    if corType=='magnet':
        data2d=expD['magnet_corr']
        vmax=1.01; vmin=-1.01
        dLab='b)'
        
    img = ax.imshow(data2d, cmap='bwr', vmin=vmin, vmax=vmax)

    cbar_ax = fig.add_axes([ax.get_position().x0+0.04, ax.get_position().y1 + 0.05, ax.get_position().width*0.8, 0.02])
    fig.colorbar(img, cax=cbar_ax, orientation='horizontal')

    ax.set(xlabel="atom index")
    ax.text(-0.06, 1.05, dLab, transform=ax.transAxes)#,c='b')#, fontsize=14)
    
    atomTicks=[j for j in range(0,nAtom,10)]
    ax.set_xticks(atomTicks)
    ax.set_yticks(atomTicks)



#=================================
#  M A I N
#=================================
if __name__ == "__main__":
    roys_fontset(plt)
    fig, axs = plt.subplots(1,2, figsize=(10,5))
      
    inpPath='/dataVault/dataQuEra_2023paper_confA/post'
    outPath='out/'

    expD0,expMD0=read4_data_hdf5(os.path.join(inpPath,'zurek_qpu_At3.0o58a.z2ph.h5'))

    plot_image_with_top_colorbar(expD0,expMD0,axs[0],corType='density')
    plot_image_with_top_colorbar(expD0,expMD0,axs[1],corType='magnet')

    # Hide x-axis labels for axr1 and axr2 to avoid label overlapping
    #plt.setp(axs[1].get_yticklabels(), visible=False)

    # Adjust figure spacing to remove white space at the bottom
    fig.subplots_adjust(bottom=0.05, wspace=0.15)

    outF='out/correlation_4top.pdf'
    fig.savefig(outF)
    print('M: saved',outF)
