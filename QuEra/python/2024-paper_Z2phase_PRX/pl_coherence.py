#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.PlotterBackbone import roys_fontset

#...!...!....................
def harvest_slope_data():
    
    trSt="0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.5 1.7 1.9 2.1 2.4 2.7 3.0"
    #trSt="0.3 0.4 0.5"

    # Convert each substring to a float
    trL = [float(num) for num in trSt.split()]
    nSamp=len(trL)
    print(trL)

    dataD={}
    for j in range(nSamp):
        xL=[]
        for c in "abcd":
            inpF='zurek_qpu_At%.1fo58%c.z2ph.h5'%(trL[j],c)
            expD,expMD=read4_data_hdf5(os.path.join(inpPath,inpF),verb=(c=='a') *( j==0))
            fmd=expMD['2pt_corr_fit']
            a,b=fmd['a'],fmd['b']
            xL.append(-1/b)
        dataD[trL[j]]=xL

    #pprint(dataD)
    return dataD


def plot_means_with_error_bars(ax, data_dict,logY=False):
    ax.set(xlabel=r"ramp time ($\mu$s)",ylabel="coherence length (sites)")
    ax.text(-0.0, 1.05, 'd)', transform=ax.transAxes)

    # Compute mean and standard deviation for each key in the dictionary
    for key, values in data_dict.items():
        mean = np.mean(values)
        std_dev = np.std(values)

        # Plot the mean with error bar
        ax.errorbar(key, mean, yerr=std_dev, fmt='o', capsize=5, color='black')
        
    # Set the x-axis limits and ticks
    ax.set_ylim(1, )
    ax.set_xlim(0.2, 5)
    ax.set_xscale('log')
    ax.set_xticks([0.5, 1, 2.,3.])
    ax.set_xticklabels(['0.5', '1', '2', '3'])

    if logY:
        ax.set_yscale('log')
        ax.set_ylim(1.5, )
        ax.set_yticks([2,3,4,6])
        ax.set_yticklabels(['2', '3', '4','6'])
    
#...!...!....................    
def XXcreate_violin_plot(ax, data_dict):
    ax.set(xlabel=r"ramp time $\mu$s",ylabel="coherence length (sites)")
    ax.text(-0.0, 1.05, 'd)', transform=ax.transAxes)

    # Prepare data for violin plot
    positions = list(data_dict.keys())
    
    data = [data_dict[key] for key in positions]

    
    # Create violin plot
    violin_parts = ax.violinplot(data, positions, widths=0.03, showmeans=False, showmedians=True)

    #violin_parts = ax.violinplot(data, positions, widths=0.03, showmeans=False, showmedians=True, showextrema=False)

    # Customize the violin plot colors if desired
    for pc in violin_parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_edgecolor('none')
        pc.set_alpha(0.7)
    
   
    # Set the x-axis limits and ticks
    ax.set_xlim(0.2, 3.1)
    ax.set_xticks([0.5, 1, 2.,3.])
    ax.set_xticklabels(['0.5', '1', '2', '3'])
    ax.set_xscale('log')
    
    
    

#...!...!....................
def plot_2pt_correl(ax):
    ax.set(xlabel="displacement (sites)",ylabel="2pt-point correlation")
    ax.text(-0.0, 1.05, 'c)', transform=ax.transAxes)
    
    timeRL=[0.3,0.5,1.5,3.0]
    timeRL.reverse()
    nSamp=len(timeRL)

    mySty=[ 'd','^','s','o' ]
    dCol='k'
    xmax=7.5
    for i in range(nSamp):
        inpF='zurek_qpu_At%.1fo58aM4.z2ph.h5'%timeRL[i]
        expD,expMD=read4_data_hdf5(os.path.join(inpPath,inpF),verb=i==0)

        XY=expD['magnet_2pt_corr']
        dMr=mySty[i]
        dLab=r"%.1f ($\mu$s)"%timeRL[i]
        X= XY[:,0]
        Y = XY[:,1]
        ax.scatter(X,Y, marker=dMr, label=dLab,c=dCol)

        #... get fit results
        XYf=expD['2pt_corr_fit']
        Xf = XYf[0]
        Yf = XYf[1]
        ax.plot(Xf,Yf,'-',c=dCol)

        #... extend fit
        fmd=expMD['2pt_corr_fit']
        a,b=fmd['a'],fmd['b']
        Yextr=a*np.exp(b*X)
        ax.plot(X,Yextr,'--',c=dCol)
        
    ax.set_yscale('log')
    ax.legend(title='ramp time')
    ax.set_xlim(0.5,xmax)
    ax.set_ylim(8e-2,1)
    ax.set_yticks([0.1, 0.5, 1.], ['0.1', '0.5', '1'])
    

#=================================
#  M A I N
#=================================
if __name__ == "__main__":
    roys_fontset(plt)
    fig, axs = plt.subplots(1,3, figsize=(14,4))
      
    inpPath='/dataVault/dataQuEra_2023paper_confA/post'
    outPath='out/'

    plot_2pt_correl(axs[0])
    slopeD=harvest_slope_data()
    #create_violin_plot(axs[1],slopeD)
    plot_means_with_error_bars(axs[1],slopeD)

    plot_means_with_error_bars(axs[2],slopeD,logY=True)

    # Adjust figure spacing to add a 20% gap between the subplots
    fig.subplots_adjust(wspace=0.25,bottom=0.2)
    
    #plt.tight_layout()
    outF='out/coherence_4bot.pdf'
    fig.savefig(outF)
    print('M: saved',outF)
