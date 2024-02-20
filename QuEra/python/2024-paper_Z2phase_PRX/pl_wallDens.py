#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.PlotterBackbone import roys_fontset
import os, sys
sys.path.append('../2024-paper_zurek')
from fit_asIs_zurek import zurek_observables_one

#...!...!....................
def harvest_wallDens_data():
    
    trSt="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.5 1.7 1.9 2.1 2.4 2.7 3.0"
    #trSt="0.3 0.4 0.5"

    # Convert each substring to a float
    trL = [float(num) for num in trSt.split()]
    nSamp=len(trL)
    print(trL)

    dataY=[]
    for j in range(nSamp):
        xL=[]        
        inpF='zurek_qpu_At%.1fo58aM4.z2ph.h5'%(trL[j])
        expD,expMD=read4_data_hdf5(os.path.join(inpPath,inpF),verb= j==0)
        wallDens,densMom,wallWidth=zurek_observables_one(expD,expMD)
        print('www densMom:',densMom)  #  [ mean_x,mean_err, std_x,tshot]        
        #dataY.append(densMom[:2] )  # mean_x,mean_err
        dataY.append([densMom[0],densMom[2]] )  # mean_x,std_x

    pprint(dataY)
    return np.array(trL),np.array(dataY)

#...!...!....................
def plot_means_with_error_bars(ax, X,Y):
    print('yy',Y.shape)
    ax.set(xlabel=r"ramp time ($\mu$s)",ylabel="Avr wall density number")
    ax.text(-0.0, 1.05, 'b)', transform=ax.transAxes)

    ax.errorbar(X,Y[:,0], yerr=Y[:,1], fmt='o', capsize=5, color='black')
    #ax.plot(X,Y[:,0],'o', color='black')
    
    #ax.set_yscale('log')
    #ax.tick_params(axis='y', which='both', labelleft=False)
    #ax.set_yticks(      [5, 6, 10, 20])
    #ax.set_yticklabels(['5','', '10', '20'])
    #ax.set_yticks(      [5, 10, 15, 20])
    #ax.set_yticklabels(['5','10', '15', '20'])
    
    ax.set_ylim(0, )
    
    ax.set_xlim(0.07, 4.5)
    ax.set_xscale('log')
    ax.set_xticks([0.1,0.5, 1, 2.,3.])
    ax.set_xticklabels(['0.1','0.5', '1', '2', '3'])

    
    

#...!...!....................
def plot_wallDensity(ax):
    ax.set(xlabel="wall density number",ylabel="probability")
    ax.text(-0.0, 1.05, 'a)', transform=ax.transAxes)
    
    timeRL=[0.3,0.5,1.5,3.0]
    timeRL=[0.1,0.2,0.3,1.0]
    timeRL.reverse()
    nSamp=len(timeRL)

    mySty=[ 'd','^','s','o' ]
    myCol=['dodgerblue','magenta','darkorange','cyan']
   
    for i in range(nSamp):
        inpF='zurek_qpu_At%.1fo58aM4.z2ph.h5'%timeRL[i]
        expD,expMD=read4_data_hdf5(os.path.join(inpPath,inpF),verb=i==0)
        wallDens,densMom,wallWidth=zurek_observables_one(expD,expMD)
        #print('wallWidth:',wallWidth)
        #print('www densMom:',densMom)  #  [ mean_x,mean_err, std_x,tshot]
        wallDens[1::2]=wallDens[::2][:-1]
        # Prepare x values for the step plot
        x_values = np.arange(len(wallDens) + 1)
        
        # Plotting the filled step function
        dCol=myCol[i]
        dLab=r"%.1f ($\mu$s)"%timeRL[i]

        ax.fill_between(x_values[:-1], wallDens, step='mid', alpha=0.55,color=dCol, label=dLab)
        
       
    ax.legend(title='ramp time')   
    ax.set_ylim(0,)
    ax.set_yticks([0, 0.1,  0.2], ['0','0.1', '0.2'])
    ax.set_xlim(0.0, 49)


#=================================
#  M A I N
#=================================
if __name__ == "__main__":
    roys_fontset(plt)
    fig, axs = plt.subplots(1,2, figsize=(10,4))
      
    inpPath='/dataVault/dataQuEra_2023paper_confA/post'
    outPath='out/'

    plot_wallDensity(axs[0])
    timeL,wallL=harvest_wallDens_data()
    
    plot_means_with_error_bars(axs[1],timeL,wallL)

    # Adjust figure spacing to add a 20% gap between the subplots
    fig.subplots_adjust(wspace=0.25,bottom=0.2)
    
    #plt.tight_layout()
    outF='out/wallDensity_f5.pdf'
    fig.savefig(outF)
