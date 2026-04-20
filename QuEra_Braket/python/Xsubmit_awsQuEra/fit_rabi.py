#!/usr/bin/env python3
'''
Single qubit Rabi vs. time

INPUT prob[PEY,NS,NB] , where
         PEY=3 for: prob, probErr, yield
         NS=  num rabi widths 
         NB=2  # states of 1 qubit 

use case:
 XX./ana_rabi2q.py   --basePath  /home/balewski/dataXtalk/ --expName   rabi2q_Q5Q0_fsfque -X
'''

import os
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.Util_Fitter  import  fit_data_model
from toolbox.PlotterBackbone import PlotterBackbone
from pprint import pprint
import copy
import numpy as np

from toolbox.ModelSinMit   import  ModelSinMit
from toolbox.ModelSin   import  ModelSin
from toolbox.ModelSinExp   import  ModelSinExp

#?class DataEmpty:    pass  # empty class

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',action='store_true', default=False,help="disable X-term for batch mode")

    parser.add_argument(  "--expName", default='rabiJune6', help="measurement name [.h5]")
    #?parser.add_argument('--readErrEps', default=None, type=float, help='probability of state 1 to be measured as state 0')

    parser.add_argument("--basePath",help="name of RB-sequence",
                        default='/dataVault/dataQuEra_2023may1AtA_qpu'
                        )

    args = parser.parse_args()
    # make arguments  more flexible
    #os.path.join(args.basePath,'meas')
    #?args.outPath=os.path.join(args.basePath,'ana')
    args.calibPath=os.path.join(args.basePath,'calib')
    args.dataPath=args.calibPath # ???
    args.outPath=args.calibPath # ???

    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.calibPath)
    return args


#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)
        self.mL7=['*','^','x','h','o','x','D']
        
#...!...!..................
    def rabi_fit(self,bigD,plDD,figId=1):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,7))
        nrow,ncol=3,1
        #rdQ=plDD['qubit_a']
        #otQ=plDD['qubit_b']

        dataTime=bigD['evolTime_us']
        dataPop=bigD['probsSum'][...,1]

        # check min/max of prob
        txtMM=''
        probMin=np.min(dataPop[0]) 
        probMax=np.max(dataPop[0])
        if probMin<0 : txtMM+=' *** MIN! '
        if probMax>1 : txtMM+=' *** MAX! '
        print('Pl min/max:',probMin,probMax, txtMM)
        
        # ------ data+fit 
        yLab='r-population'
        xLab='Rabi Hamiltonian duration (us),     meas='+plDD['short_name']
        
        tit=plDD['exp_comment']
        print('PRF:tit',tit)

        ax = self.plt.subplot2grid((nrow,ncol), (0,0), colspan=1, rowspan=2 )
        #print('P:rf dataPop',dataPop.shape)

        dLab='meas '
        dCol='b'

        dFmt='o'
        ax.errorbar(dataTime,dataPop[0],yerr=dataPop[1],fmt=dFmt,color=dCol,markersize=4,linewidth=0.8,label=dLab)
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))
            
        # add fit to ax=Z
        ftag='fit_sin'
        fV=plDD[ftag+'_pred']
        ax.plot(dataTime,fV,color='r',label=ftag)
        fitMD=plDD[ftag]

        CV=fitMD['fit_result']['fitPar']['C'] 
        yAvr=CV[0]
        ax.axhline(yAvr,linestyle='--',linewidth=1,color='r')
        ax.axhline(0.5,linestyle='--',linewidth=0.8,color='k')
        
        FV=fitMD['fit_result']['fitPar']['F']
        redchi=fitMD['fit_result']['fitQA']['redchi'][0]        
        txt1='freq/MHz: %.3f +/- %.3f  chi2/ndf=%.1f'%(FV[0], FV[1],redchi)
        print(ftag,txt1)
        ax.text(0.44,0.9,txt1,transform=ax.transAxes,color=dCol,fontsize=12)

        spTxt='SPAM corr=none'
        #?if plDD['SPAM_corrected']!='none':
        #?    spTxt='SPAM corr=%s, detM=%.2f'%(plDD['SPAM_corrected'],plDD['SPAM_info']['det_M'])
        #?ax.text(0.64,0.83,spTxt,transform=ax.transAxes,color=dCol,fontsize=12)
        
        ax.set_ylim(-0.05,1.05)
        #ax.set_xlim(-5.,)
        ax.grid()
        
        ax.set( ylabel=yLab, title=tit)
        ax.legend(loc='center')

        #....... residua .....
        dataPop[0]-=fV
        ax = self.plt.subplot(nrow,ncol,3)
        dt1=dataTime[1]-dataTime[0]
        ax.errorbar(dataTime,dataPop[0],yerr=dataPop[1],fmt="*",color='r',markersize=2,linewidth=0.8)
        ax.bar(dataTime,dataPop[0], width=0.8*dt1)
        ax.set(xlabel=xLab,ylabel='meas-fit')
        ax.grid()
        if 'timeLR' in plDD:  ax.set_xlim(tuple(plDD['timeLR']))


        
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()
    inpF=args.expName+'.ags.h5'
    bigD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))
    #XM_print_nice_calib_csv()
    print('M:expMD:');  pprint(expMD)
    
    expMD['short_name']='fit_'+expMD['short_name']
                                
    dataPop=bigD['probsSum'][...,1]
    twidth_us=bigD['evolTime_us']
    print('M:dataPop',dataPop.shape)
    print('twidth_us',twidth_us)
    
    # ...... fitting rabi-sine wave 
    oscModel=ModelSinExp()
    #if not PLOT['fitSin']: break  # for debugging

    fitY,fitMD=fit_data_model(dataPop,twidth_us,oscModel)
    ftag='fit_sin'
    bigD['fit_meas']=fitY.astype('float32')
    expMD[ftag]=fitMD       
    
    if args. verb>1: pprint(fitMD)

    #...... WRITE  OUTPUT 
    outF=os.path.join(args.calibPath,expMD['short_name']+'.fit.h5')
    write4_data_hdf5(bigD,outF,expMD)
    
    #--------------------------------
    # ....  plotting ........
    args.prjName=expMD['short_name']
    #?if anaMD['SPAM_corrected']!='none': args.prjName+='_'+anaMD['SPAM_corrected'] 

    plot=Plotter(args)
    plDD=copy.deepcopy(expMD)
    plDD[ftag+'_pred']=fitY
    plDD[ftag]=fitMD
    plDD['exp_comment']='rabi exp, '+expMD['short_name']+' exec: '+expMD['job_qa']['exec_date']
    #print('M:plDD:');pprint(plDD)
    plDD['timeLR']=[0.,4.1]  # (ns)  time range clip
    plot.rabi_fit(bigD,plDD)

    plot.display_all()
    print('M:done')

