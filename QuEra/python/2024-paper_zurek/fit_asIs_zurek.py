#!/usr/bin/env python3
""" 
analyze 58-atom loop data for Kibble-Zurek scaling
Data taken on Sept-26

11 Aquila    jobs
   for t_de in 0.1 0.2 0.3 0.4 0.5 0.7 1.0 1.3 1.7 2.1 3.0 ; do
       t_us=t_de+1.0
       submit  … --chain_shape  58  --evol_time_us  $t_us

Laptop data location: /dataVault/dataQuEra_2023paper_qpu/ana/
['zurek_qpu_td0.1o58a.z2ph.h5', 'zurek_qpu_td1.3o58a.z2ph.h5', 'zurek_qpu_td0.4o58a.z2ph.h5'] ...

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import  time
import sys,os
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.PlotterBackbone import PlotterBackbone
from bitstring import BitArray

# fitting tau_Q
from toolbox.Util_Fitter  import  fit_data_model
from toolbox.ModelPower import ModelPower
from matplotlib.ticker import ScalarFormatter  
import re  # for count_wall_depth(.)

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "--dataPath", default='/dataVault/dataQuEra_2023paper_confB/post/',help="input path ")
    parser.add_argument("-o", "--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("--groupName", default='o58b',help="group of experiments with common purpose")
    
    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-p", "--showPlots",  default='dmwf', nargs='+',help="abc-string listing shown plots")


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
    def domain_wall_density(self,md,bigD,figId=1):
        amd=md['postproc']
        #pd=md['payload']
        nAtom= amd['num_atom']
         
        dens2D=bigD["domain_wall_dens"]
        timeV=bigD["detune_time_us"]
        momentV=bigD["wall_dens_moments"]
        k=timeV.shape[0]
        ncol=2
        nrow=int(np.ceil(k/ncol))
        print('ncn',k,nrow,ncol)
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,12))        
        # Create subplots within the specified figure
        #    ax, ax2 = fig.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
        
        nAtom=md['postproc']['num_atom']
        # Generate the values of X corresponding to the distribution
        densX = np.arange(dens2D.shape[1])
        dFmt='o'
        
        for i in range(k):
            ax = self.plt.subplot(nrow,ncol,1+i)
            # Create a bar plot to visualize the histogram
            #print('hh', densX.shape, dens2D[i].shape)
            ax.bar(densX,dens2D[i] , width=1., edgecolor='black', color='lightgreen')
            tLab='t_detune=%.1f (us), %s'%(timeV[i],md['sum_title'])
            ax.set(xlabel="wall density number",ylabel="probability",title=tLab)

            #  beautification
            avrX,avrXer,stdX,mshots=momentV[i]
            yy=0.1
            ax.errorbar([avrX], [yy] , xerr=avrXer,fmt=dFmt,color='r',markersize=4,linewidth=0.8)
            txt1='avr %.2f +/-%.2f  \nstd=%.1f  shots=%d'%(avrX,avrXer,stdX,mshots)
            ax.text(20,yy,txt1,color='b')
            if 'densLR' in md:   ax.set_xlim(tuple(md['densLR']))
            ax.grid()
            
            #ax.errorbar(densX, dens2DdataTime,dataPop[0],yerr=dataPop[1],fmt=dFmt,color=dCol,markersize=4,linewidth=0.8,label=dLab)
            #break
        return

        #.... mis text
        txt1='device: '+md['submit']['backend']
        txt1+='\nexec: %s'%md['job_qa']['exec_date']
        txt1+='\ndist %s um'%(md['payload']['atom_dist_um'])
        txt1+='\nOmaga %.1f MHz'%md['payload']['rabi_omega_MHz']
        txt1+='\nshots/job: %d'%md['submit']['num_shots']
        txt1+='\ntot atoms: %d'%md['payload']['tot_num_atom']
        txt1+='\nnum jobs: %d'%nTime
        txt1+='\nreadErr eps: %s'%md['postproc']['readErr_eps']
        ax.text(0.02,0.55,txt1,transform=ax.transAxes,color='g')

        return
    
    #...!...!....................
    def wall_density_vs_tauQ(self,md,bigD,figId=2):
        figId=self.smart_append(figId)
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,6))
       
        smd=md['submit']
        amd=md['postproc']
        pmd=md['payload']
        nAtom= amd['num_atom']
        detuneChange=pmd['detune_change_MHz']
        
        #nominal_Rb= pd['nominal_Rb_um']
        timeV=bigD["detune_time_us"]
        momentV=bigD["wall_dens_moments"]
        #k=timeV.shape[0]

        tauV=detuneChange/timeV  # MHz/us
        
        # .... raw data
        ax = self.plt.subplot(nrow,ncol,1)
        ax.errorbar(timeV, momentV[:,0] , yerr= momentV[:,1],fmt='s',color='g',markersize=4,linewidth=0.8)
        tLab='raw data  %s'%(md['sum_title'])
        ax.set(ylabel="avr wall density number",xlabel="t_detune (us)",title=tLab)
        ax.set_ylim(0,)
        ax.grid()
        
        # ....  money plot
        ax = self.plt.subplot(nrow,ncol,2 )
        ax.errorbar(1/tauV, momentV[:,0] , yerr= momentV[:,1],fmt='o',color='b',markersize=4,linewidth=0.8)
        tLab='money plot1  %s'%(md['sum_title'])
        ax.set(ylabel="avr wall density number",xlabel="1/tau_Q (us/MHz)",title=tLab)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
        
    
    #...!...!....................
    def money_fit(self,md,bigD,figId=2):
        figId=self.smart_append(figId)
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,6))
       
        smd=md['submit']
        amd=md['postproc']
        pmd=md['payload']
        fmd=md['lmfit']['fit_result']
        nAtom= amd['num_atom']
        
        momentV=bigD["wall_dens_moments"]
        tauV=bigD['tau_Q'] # MHz/us
        #print('sss', momentV.shape , tauV.shape)
        assert momentV.shape[0] == tauV.shape[0]


        # ------ all data
        yLab="avr wall density number"
        xLab=r"$\tau_Q ~(MHz/\mu s)$"
        tLab='money2  %s'%(md['short_name'])
        
        ax = self.plt.subplot(nrow,ncol,1 )
        dCol='b'
        dLab='Aquila meas'
        ax.errorbar(tauV, momentV[:,0] , yerr= momentV[:,1],fmt='o',color=dCol,markersize=4,linewidth=0.8,label=dLab)

        #  fit data + fit func
        dataX=bigD['lmfit_x'] # MHz/us
        dataP=bigD['lmfit_y']  # <n>  PE
        fitY= bigD['lmfit_f']   # <n> : avr wall density
        dCol='r'
        dLab=r'fit: $a \cdot (\tau_Q)^{\mu}$'
        ax.plot(dataX,fitY,'-',label=dLab,color=dCol)

        pprint(fmd)
        # ... beautification
        f_mu=fmd['fitPar']['MU']
        redchi=fmd['fitQA']['redchi'][0]        
        txt1=r'$\mu =  %.3f\pm  %.3f$'%(f_mu[0], f_mu[1])
        print(txt1)
        ax.text(0.1,0.7,txt1,transform=ax.transAxes,color=dCol,fontsize=10)
        txt1='chi2/ndf=%.1f'%(redchi)
        print(txt1)
        ax.text(0.1,0.64 ,txt1,transform=ax.transAxes,color=dCol,fontsize=10)
        
        ax.set(ylabel=yLab,xlabel=xLab,title=tLab)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
        ax.legend(loc='upper left')

        # Customize y-axis tick labels to exclude "10^0"
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        self.plt.gca().yaxis.set_major_formatter(formatter)
        self.plt.gca().xaxis.set_major_formatter(formatter)

        # Set the y-axis ticks to display at 1, 5, and 10
        ax.set_yticks([ 6, 8, 10, 15, 20])
        ax.set_xticks([ 2, 5, 10, 20, 50])
 
    #...!...!....................
    def wall_width(self,md,bigD,figId=1):
        timeV=bigD["detune_time_us"]
        wwidth2D=bigD["wall_width"]  
        widthX = np.arange(wwidth2D.shape[1])+1  # counts form 2
        k=wwidth2D.shape[0]
        ncol=2
        nrow=int(np.ceil(k/ncol))
        print('ncn2',k,nrow,ncol)
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,12))        
        
        for i in range(k):
            ax = self.plt.subplot(nrow,ncol,1+i)
            # Create a bar plot to visualize the histogram
            #print('hh', densX.shape, dens2D[i].shape)
            ax.bar(widthX,wwidth2D[i] , width=1., edgecolor='black', color='lightblue')
            tLab='t_detune=%.1f (us), %s'%(timeV[i],md['sum_title'])
            ax.set(xlabel="wall width",ylabel="counts per shot",title=tLab)
            # Set the x-axis ticks to show only 4 ticks
            ax.set_xticks(widthX)

            frac=(wwidth2D[i,0]+wwidth2D[i,1])/np.sum(wwidth2D[i])
            
            txt='/- -%.0f%c - - \\'%(100*frac,37)
            ax.text(0.15,0.55,txt,transform=ax.transAxes,color='r',fontsize=12)


#............................
#.......... CLASS END .......
#............................

  
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


#...!...!....................
def compute_moments(probV,tshot):
    # INPUT: numpy array representing the probability distribution

    sum_p= np.sum( probV)
    # Generate the values of X corresponding to the distribution
    x_values = np.arange(len(probV))

    # Compute the mean (expected value) of X distribution
    mean_x = np.sum(x_values * probV)/sum_p
        
    # Compute the standard deviation of X
    std_x = np.sqrt(np.sum((x_values - mean_x)**2 * probV)/sum_p )
    mean_err= std_x/np.sqrt(tshot)

    print("sum prob =%.1f  mean X=%.2f +/-%.2f,  std X=%.1f"%( sum_p,mean_x,mean_err, std_x))
    #print("sum prob =%.1f  mean X=%.1f,  var X=%.1f"%( sum_p,mean_x, std_x**2))
    return [ mean_x,mean_err, std_x,tshot]
    
#...!...!....................
def zurek_observables_one(bigD,md):
    amd=md['postproc']
    #pmd=md['payload']
    nAtom= amd['num_atom']
    hexpattV=bigD['ranked_hexpatt']
    mshotV=bigD['ranked_counts']
    nSol=hexpattV.shape[0]
    print('zurek_observables START, numSol=%d'%(nSol))    

    cntH=[0 for i in range(nAtom+1) ]
    tshot=0
    w=4 # maximal wall width
    wwidthH=[0 for i in range(w)]  # accumulator

    for k in range(nSol):
        mshot=mshotV[k]
        hexpatt=hexpattV[k].decode("utf-8")
        A=BitArray(hex=hexpatt)[-nAtom:]  # clip leading 0s
        binpatt=A.bin
        if nAtom==58:  binpatt=binpatt+binpatt[0]
        xV = np.array(list(binpatt))
        # Count all occurrences of '11' or '00' patterns
        n11 = np.sum(np.logical_and(xV[:-1] == '1', xV[1:] == '1'))
        n00 = np.sum(np.logical_and(xV[:-1] == '0', xV[1:] == '0'))
        nw=n00+n11
        
        #print('\nhex:',hexpatt,nAtom,binpatt, len(binpatt),'nw=',nw,mshot)
        cntH[nw]+=mshot
        tshot+=mshot
        count_wall_depth(binpatt,wwidthH,mshot)
        #break    
    wwidthV=np.array(wwidthH)/tshot
    wallDensV=np.array(cntH)/tshot
    #print(' wallDensV dump:', wallDensV, 'tshot=',tshot)
    wallMoments=compute_moments(wallDensV,tshot)
    print('wwidthV:',wwidthV)
    
    return  wallDensV,wallMoments,wwidthV

#...!...!....................
def count_wall_depth(bits,wwidthH,mshot):
    #bits='1011101110010000100000'
    w=len(wwidthH)
    #print('CWD   w=%d'%(w)); print('bits:',bits)
    #  find all sequences of consecutive 1s in the binary string x.
    #  The regular expression r'1+' matches one or more consecutive 1s.

    L0=re.findall(r'0+', bits)
    L1=re.findall(r'1+', bits)
    #print('L0=',L0,'\nL1=',L1)
    
    # counts form 1
    for x in L0: wwidthH[ min(w,len(x))-1 ]+=mshot
    for x in L1: wwidthH[ min(w,len(x))-1 ]+=mshot
    return
    
#...!...!....................
def fit_tauQ(bigD,md):
    pmd=md['payload']
    amd=md['postproc']
    nAtom= amd['num_atom']
    detuneChange=pmd['detune_change_MHz']
    timeV=bigD["detune_time_us"]
    momentV=bigD["wall_dens_moments"]
    #momentV[:,0]=[19.76,9.28, 5.12,3.48,3.4,2.8,2.36,1.69,1.59,1.52,1.82] # manual ZNE for readout error
    
    tauV=detuneChange/timeV  # MHz/us
    #print('dd',tauV.dtype)

    #idxL= np.logical_and(tauV> 4,tauV<25)
    idxL= np.logical_and(tauV> 2.5,tauV<20) # confA
    idxL= np.logical_and(tauV> 4.5,tauV<25) # confB
    # ...... fitting rabi-sine wave
    oscModel=ModelPower()
    #if not PLOT['fitSin']: break  # for debugging

    # expects PEY=2+  for: prob, probErr, anything else
    dataP=  np.swapaxes(momentV[idxL],0,1)  # [PEY,nTime]
    dataX=tauV[idxL]
    print('inp t_det:',timeV[idxL])
    print('fit x-values:',dataX )
    print('fit input y-values:', dataP[0])
    print('fit input yerr-values:', dataP[1])
    fitY,fitMD,_=fit_data_model(dataP,dataX,oscModel)
    print('fit output y-values:', fitY)
    # store input & fit results
    bigD['tau_Q']=tauV.astype('float32')  # MHz/us
    bigD['lmfit_x']=dataX.astype('float32')  # MHz/us
    bigD['lmfit_y']=dataP[:2].astype('float32')  # <n>  PE
    bigD['lmfit_f']=fitY.astype('float32')   # <n> : avr wall density
    md['lmfit']=fitMD

    
    
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()
    #coreN='o58a_h0'  # to split data on halves
           
    sumFL=locateSummaryData( args.dataPath,pattern=args.groupName) # 'c17a') #
    assert len(sumFL) >0
    
    #sumFL=sumFL[8:9] #tmp, for testing 'o58a')
          
    detuneTimeV=[]
    wallDensV=[]   # distribution of wall density
    densMomV=[] # moments of density
    wallWidthV=[]
    for i,inpF in enumerate(sumFL):        
        expD,expMD=read4_data_hdf5(os.path.join( args.dataPath,inpF),verb=i==0)
        if i==0: pprint(expMD)
        evolTime=expMD['payload'].pop('evol_time_us')
        ramp2T=sum(expMD['payload']['rabi_ramp_time_us'])
        detT=evolTime-ramp2T
        
        #print('\nttt',evolTime, ramp2T,detT)
    
        detuneTimeV.append(detT)
        wallDens,densMom,wallWidth=zurek_observables_one(expD,expMD)
        wallDensV.append(wallDens)
        densMomV.append(densMom)
        wallWidthV.append(wallWidth)

    # convert to numpy
    detuneTimeV=np.array(detuneTimeV) # shape [nTime]
    wallDensV=np.array(wallDensV) # shape [nTime, wallCount]
    wallWidthV=np.array(wallWidthV)
    densMomV=np.array(densMomV) # shape [nTime, DP=4], (mean, meanErr, std, shots)
    
    print('M:wallDensV:',wallDensV.shape,wallWidthV.shape)
    
    # update meta data
    txt1='oct-2' if '20231002' in expMD['submit']['date']  else 'sept-26' 
    expMD['short_name']= '%s.m%d_%s'%(args.groupName,len(sumFL),txt1)
    expMD['sum_title']='58-atoms sept-26 ver1'
    a,b=expMD['payload']['detune_delta_MHz']
    expMD['payload']['detune_change_MHz']=b-a
    bigD={}
    bigD["detune_time_us"]=detuneTimeV
    bigD["domain_wall_dens"]=wallDensV
    bigD["wall_dens_moments"]=densMomV
    bigD["wall_width"]=wallWidthV
    #

    #... more  post processing
    fit_tauQ(bigD,expMD)
    
    # ----  just plotting
    args.prjName=expMD['short_name']
    plot=Plotter(args)
    if 'd' in args.showPlots:
        expMD['densLR']=[0.,40.]  
        plot.domain_wall_density(expMD,bigD,figId=1)
    if 'm' in args.showPlots:
        plot.wall_width(expMD,bigD,figId=10)

    if 'w' in args.showPlots:
        plot.wall_density_vs_tauQ(expMD,bigD,figId=2)
    if 'f' in args.showPlots:
        plot.money_fit(expMD,bigD,figId=3)

    if 'axx' in args.showPlots:
        ax2=plot.scar_evolution(expMD,bigD,figId=22)
        from ProblemZ2Phase import ProblemZ2Phase
        task= ProblemZ2Phase(args,expMD,expD)
        expMD['TrangeLR']=[0.,4.]  # (um)  x-range range clip
        plot.global_drive(task,axL=[ax2,None])       

    plot.display_all(png=1)
    

