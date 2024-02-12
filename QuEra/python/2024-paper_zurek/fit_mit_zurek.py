#!/usr/bin/env python3
""" 
analyze 58-atom loop data for Kibble-Zurek scaling
Data taken on Oct-2

Kibble-Zurek mechanism analysis

 Merge mutiple jobs for the same circuit
 apply SPAM mitigation sing code from Milo

Laptop data location: /dataVault/dataQuEra_2023paper_confB


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
from apply_mitWall import wall_count_mitig_v1
from toolbox.UAwsQuEra_job import flatten_ranked_hexpatt_array

# fitting tau_Q
from toolbox.Util_Fitter  import  fit_data_model
from toolbox.ModelPower import ModelPower
from matplotlib.ticker import ScalarFormatter  


import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("--expConf",default='B',help="name of a group of experiments")

    parser.add_argument( "--readoutError",  default=[0.01, 0.08] , type=float, nargs='+', help="readout error for [ set0meas1, set1meas0 ]")
 
    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-p", "--showPlots",  default='r m f g', nargs='+',help="abc-string listing shown plots")


    args = parser.parse_args()
    args.showPlots=''.join(args.showPlots)
    args.basePath='/dataVault/dataQuEra_2023paper_conf'+args.expConf
    args.dataPath=os.path.join(args.basePath,'post')
    args.outPath=os.path.join(args.basePath,'remit')

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert  args.expConf in ['A','B','C']    
    assert os.path.exists(args.outPath)

    return args

#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)
    
    #...!...!....................
    def wall_density_vs_tauQ(self,md,bigD,tag,figId=2):
        figId=self.smart_append(figId)
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,6))
       
        smd=md['submit']
        amd=md['postproc']
        pmd=md['payload']
        nAtom= amd['num_atom']
        
        timeV=bigD["detune_time_us"]
        tauQ=bigD["tauQ_MHz/us"]
        momentV=bigD["wall_dens_"+tag]
          
        # .... raw data
        ax = self.plt.subplot(nrow,ncol,1)
        ax.errorbar(timeV, momentV[:,0] , yerr= momentV[:,1],fmt='s',color='g',markersize=4,linewidth=0.8)
        tLab='data  %s  %s'%(tag,md['short_name'])
        ax.set(ylabel="avr wall density number",xlabel="t_detune (us)",title=tLab)
        ax.set_ylim(0,)
        ax.grid()
        
        # ....  money plot
        ax = self.plt.subplot(nrow,ncol,2 )
        ax.errorbar(1/tauQ, momentV[:,0] , yerr= momentV[:,1],fmt='o',color='b',markersize=4,linewidth=0.8)
        tLab='money1  %s  %s'%(tag,md['short_name'])
        ax.set(ylabel="avr wall density number",xlabel="1/tau_Q (us/MHz)",title=tLab)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
        
    
    #...!...!....................
    def money_fit(self,md,bigD,tag,figId=2):
        figId=self.smart_append(figId)
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(5,6))
       
        smd=md['submit']
        amd=md['postproc']
        pmd=md['payload']
      
        nAtom= amd['num_atom']
        tauQ=bigD["tauQ_MHz/us"]
  

        # ------ all data
        yLab="avr wall density number"
        xLab=r"$\tau_Q ~(MHz/\mu s)$"
        tLab='money2 %s  %s'%(tag,md['short_name'])

        #  fit data + fit func
        ftag='lmfit_'+tag
        fmd=md[ftag]['fit_result']
        momentV=bigD["wall_dens_"+tag]
        print('ss1',momentV.shape,tauQ.shape)
        assert momentV.shape[0] == tauQ.shape[0]

        
        ax = self.plt.subplot(nrow,ncol,1 )
        dCol='b'
        dLab='Aquila meas'
        ax.errorbar(tauQ, momentV[:,0] , yerr= momentV[:,1],fmt='o',color=dCol,markersize=3,linewidth=0.8,label=dLab)

 
        dataX=bigD[ftag+'_x'] # MHz/us
        dataP=bigD[ftag+'_y']  # <n>  PE
        fitY= bigD[ftag+'_f']   # <n> : avr wall density
        dCol='r'
        dLab=r'fit: $a \cdot (\tau_Q)^{\mu}$'
        ax.plot(dataX,fitY,'-',label=dLab,color=dCol)

        pprint(fmd)
        # ... beautification
        f_mu=fmd['fitPar']['MU']
        redchi=fmd['fitQA']['redchi'][0]        
        txt1=r'$\mu =  %.3f\pm  %.3f$'%(f_mu[0], f_mu[1])
        txt1+='\nchi2/ndf=%.1f'%(redchi)
        print(txt1)
        ax.text(0.05,0.66,txt1,transform=ax.transAxes,color=dCol,fontsize=10)
        txt1='SPAM s0m1,s1m0:%s'%(amd['readout_error'])
        print(txt1)
        ax.text(0.05,0.60 ,txt1,transform=ax.transAxes,color='g',fontsize=10)
        
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

        if 'densLR' in md:   ax.set_ylim(tuple(md['densLR']))

        if 0:  # Set the y-axis ticks to display at 1, 5, and 10
            ax.set_yticks([0.7, 1,2,4,6, 8, 10, 15, 20,30])
            ax.set_xticks([ 2, 5, 10, 20, 50])
        if 1:            
            ax.set_yticks([2,4, 8, 16, 32])
            ax.set_xticks([ 2, 5, 10, 20, 50])
 

#............................
#.......... CLASS END .......
#............................

  
#...!...!....................
def processOneAHS(src_dir,mySpam,tag):    
    inpF=tag

    batchL=['a','b','c','d']
    bitpattL=[]
    tshots=0
    for i,batch in enumerate(batchL):

        inpF=inpF[:-1]+batch
        expD,inpMD=read4_data_hdf5(os.path.join(args.dataPath,inpF+'.z2ph.h5'),verb=0)
        if args.verb>=2:   print('M:expMD:');  pprint(inpMD);  stop3
        
        nAtom=inpMD['payload']['num_atom']
        ishots=inpMD['postproc']['used_shots']
        tshots+=ishots
        print('add %d shots from inpF=%s'%(ishots,inpF))

        countsV=expD['ranked_counts']
        hexpattV=expD['ranked_hexpatt']
        bitpattV=flatten_ranked_hexpatt_array(countsV,hexpattV,nAtom)
        bitpattL+=bitpattV


    print('M:tot shots:',len(bitpattL),tshots)
    print('\nM:now apply SPAM correction...')
    outD=wall_count_mitig_v1(bitpattL,eps01=mySpam[0],eps10=mySpam[1])
    pprint(outD)
    '''
    print('\nrelative errors:')
    for x in sorted(outD):
        val,err=outD[x]
        print('%s  nSig=%.1f'%(x, val/err))
    '''
    return tshots,outD,inpMD
    
#...!...!....................
def fit_tauQ(bigD,md,tag):
    pmd=md['payload']
    amd=md['postproc']
    nAtom= amd['num_atom']
    detuneChange=pmd['detune_change_MHz']
    tauQ=bigD["tauQ_MHz/us"]
    #timeV=bigD["detune_time_us"]
    momentV=bigD["wall_dens_"+tag]
    
    #tauV=detuneChange/timeV  # MHz/us
    #print('dd',tauV.dtype)

    #idxL= np.logical_and(tauV> 4,tauV<25)
    #idxL= np.logical_and(tauV> 2.5,tauQ<20) # confA
    if tag=='raw':
        idxL= np.logical_and(tauQ> 4.5,tauQ<25) # confB
    if tag=='mit':
        idxL= np.logical_and(tauQ> 4.5,tauQ<30) # confB
    # ...... fitting rabi-sine wave
    oscModel=ModelPower()
    #if not PLOT['fitSin']: break  # for debugging

    # expects PEY=2+  for: prob, probErr, anything else
    dataP=  np.swapaxes(momentV[idxL],0,1)  # [PEY,nTime]
    dataX=tauQ[idxL]
    #print('inp t_det:',timeV[idxL])
    print('fit x-values:',dataX )
    print('fit input y-values:', dataP[0])
    print('fit input yerr-values:', dataP[1])
    fitY,fitMD,_=fit_data_model(dataP,dataX,oscModel)
    print('fit output y-values:', fitY)
    # store input & fit results
    #bigD['tau_Q']=tauV.astype('float32')  # MHz/us
    ftag='lmfit_'+tag
    bigD[ftag+'_x']=dataX.astype('float32')  # MHz/us
    bigD[ftag+'_y']=dataP[:2].astype('float32')  # <n>  PE
    bigD[ftag+'_f']=fitY.astype('float32')   # <n> : avr wall density
    md[ftag]=fitMD

    
#...!...!....................
def nice_dump():
    timeV=bigD["detune_time_us"]
    tauQ=bigD["tauQ_MHz/us"]
    momRaw=bigD["wall_dens_raw"]
    momMit=bigD["wall_dens_mit"]
    shotsV=bigD["used_shots"]
    print('\ndataset:', expMD['short_name'])
    print('t detune(us),tauQ (MHz/us),raw <n>,raw err <n>,raw <n>,mit err <n>,used shots')  
    nT=timeV.shape[0]
    for i in range(nT):
        print('%.1f,%.2f,%.2f,%.3f,%.2f,%.2e,%d'%(timeV[i],tauQ[i],momRaw[i][0],momRaw[i][1],momMit[i][0],momMit[i][1],shotsV[i]))
    print()
    
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()
    #coreN='o58a_h0'  # to split data on halves
    mySpam=args.readoutError
    print('\nM:assumed SPAM probability  s0m1: %.2f   s1m0: %.2f'%(mySpam[0],mySpam[1]))

    #... generate list of  distinct circuits
    tdetL=[ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.1, 2.4, 2.7, 3.0 ]

    #1tdetL=tdetL[:6]  # tmp, testing only
   
    wallDensRaw=[]
    wallDensMit=[]
    shotsV=[]
    for t_detune in tdetL :
        expTag='zurek_qpu_%st%.1fo58a'%(args.expConf,t_detune)
        print('data tag',expTag)
        tshots,tmpD,tmpMD=processOneAHS(args.dataPath,mySpam,tag=expTag)
        shotsV.append(tshots)
        wallDensRaw.append(tmpD['measured'])
        wallDensMit.append(tmpD['mitigated'])
        
      
    # convert to numpy
    detuneTimeV=np.array(tdetL) # shape [nTime]
    wallDensRaw=np.array(wallDensRaw) # shape [nTime, wallCount]
    wallDensMit=np.array(wallDensMit) # shape [nTime, wallCount]
    shotsV=np.array(shotsV)
    print('M:wallDensRaw:',wallDensRaw.shape)
    
    # update meta data
    expMD=tmpMD 
    expMD['short_name']= 'oct-2_conf%s'%args.expConf
    a,b=expMD['payload']['detune_delta_MHz']
    expMD['payload']['detune_change_MHz']=b-a
    expMD['postproc']['readout_error']=mySpam
    
    bigD={}
    bigD["detune_time_us"]=detuneTimeV
    bigD["tauQ_MHz/us"]=(b-a)/detuneTimeV
    bigD["wall_dens_raw"]=wallDensRaw
    bigD["wall_dens_mit"]=wallDensMit
    bigD["used_shots"]=shotsV

    #... more  post processing
    fit_tauQ(bigD,expMD,tag='raw')
    fit_tauQ(bigD,expMD,tag='mit')

    nice_dump()
    # ----  just plotting
    args.prjName=expMD['short_name']
    plot=Plotter(args)

    if 'r' in args.showPlots:
        plot.wall_density_vs_tauQ(expMD,bigD,tag='raw',figId=1)

    if 'm' in args.showPlots:
        plot.wall_density_vs_tauQ(expMD,bigD,tag='mit',figId=2)

    if 'f' in args.showPlots:
        plot.money_fit(expMD,bigD,tag='raw',figId=3)
        
    if 'g' in args.showPlots:
        #expMD['densLR']=[2.,28.] 
        plot.money_fit(expMD,bigD,tag='mit',figId=4)

    plot.display_all(png=1)
    

