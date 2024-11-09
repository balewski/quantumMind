#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Analyze   QCrank  experiment

'''

import os
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

from time import time
from pprint import pprint
import numpy as np
from PlotterQCrank import Plotter
from toolbox.Util_Qiskit import unpack_numpy_to_counts
from qiskit.result.utils import marginal_distribution

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3,4],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-p", "--showPlots",  default='a', nargs='+',help="abcd-string listing shown plots")
    
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")
    parser.add_argument("--basePath",default=None,help="head path for set of experiments, or 'env'")
    parser.add_argument("--inpPath",default='out/',help="raw outputs from experiment")
    parser.add_argument("--outPath",default='out',help="post-processed results ")
    parser.add_argument('-e',"--expName",  default='exp_62a21daf',help='IBMQ experiment name assigned during submission')
    
    parser.add_argument('-T',"--type",  default=0, help='Plot CudaQ or IBM results')

    args = parser.parse_args()
    # make arguments  more flexible 
    if 'env'==args.basePath: args.basePath= os.environ['EHands_dataVault']
    if args.basePath!=None:
        args.inpPath=os.path.join(args.basePath,'meas')
        args.outPath=os.path.join(args.basePath,'post')
    args.showPlots=''.join(args.showPlots)
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    return args


#...!...!....................
def marginalize_qcrank_EV(  addrBitsL, probsB,dataBit):
    #print('MQCEV inp bits:',dataBit,addrBitsL)
    # ... marginal distributions for 2 data qubits, for 1 circuit
    assert dataBit not in addrBitsL
    bitL=[dataBit]+addrBitsL
    #print('MQCEV bitL:',bitL)
    probs=marginal_distribution(probsB,bitL)

    #.... comput probabilities for each address
    nq_addr=len(addrBitsL)
    seq_len=1<<nq_addr
    mdata=np.zeros(seq_len)
    fstr='0'+str(nq_addr)+'b'
    for j in range(seq_len):
        mbit=format(j,fstr)
        mbit0=mbit+'0'; mbit1=mbit+'1'
        m1=probs[mbit1] if mbit1 in probs else 0
        m0=probs[mbit0] if mbit0 in probs else 0
        ms=m0+m1
        #print(j,mbit,'sum=',ms)        
        p=m1/ms if ms>0 else 0
        mdata[j]=p
    return 1-2*mdata
 



#...!...!.................... 
def postproc_QCrank(expD,md):
    pmd=md['payload']
    smd=md['submit']
    nq_addr=pmd['nq_addr']
    nq_data=pmd['nq_data'] 
    seq_len=pmd['num_addr']
    nImg=pmd['num_sample']
    
    countsL=unpack_numpy_to_counts(md,expD)
    
    rec_udata=np.zeros((nImg,nq_data,seq_len)) # before  re-assembling  images    
    addrBitsL = [nq_data+i  for i in range(nq_addr)]

    #print('inp data:',expD['inp_udata'])
    print('rec_udata:',rec_udata.shape,'addrBitsL:',addrBitsL)
        
    T0=time()
    for ic in range(nImg):
        counts=countsL[ic]     
        #if ic<2: print('\nPEH: ic=%d counts size'%ic,len(counts));    pprint(counts)
        for ibit in range(nq_data):
            T1=time()
            rec_udata[ic,ibit]= marginalize_qcrank_EV( addrBitsL, counts, dataBit=ibit)
            if ic<5: print(' ic=%d marginal ibit=%d  done , elaT=%.1f min'%(ic,ibit,(T1-T0)/60.))
    expD['rec_udata']=rec_udata
    #print('PEH: rec_udata:',rec_udata)
    #print('PEH: inp_udata:',expD['inp_udata'])
    
        
    
#...!...!.................... 
def residual_ana(expD,md):
    rdata=expD['rec_udata'].flatten()
    tdata=expD['true_out_udata'].flatten()  # im,add,dat
    res_data = rdata - tdata
    mean = np.mean(res_data)
    std = np.std(res_data)
    # assuming normal distribution, compute std error of std estimator
    # SE_s=std/sqrt(2(n-1)), where n is number of samples
    N=res_data.shape[0]
    se_s=std/np.sqrt(2*(N-1))
    pom=md['postproc']
    pom['res_mean']=float(mean)
    pom['res_std']=float(std)
    pom['res_SE_s']=float(se_s)
  
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)
                    
    inpF=args.expName+'.meas.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.inpPath,inpF))
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        exit(55)

    postproc_QCrank(expD,expMD)

      
    #...... WRITE  OUTPUT
    outF=os.path.join(args.outPath,expMD['short_name']+'.post.h5')
    write4_data_hdf5(expD,outF,expMD)
    
    #--------------------------------
    # ....  plotting ........
    args.prjName=expMD['short_name']
    expMD['plot']={'resid_max_range':0.4}

    plot=Plotter(args)
    
    if 'a' in args.showPlots:
        plot.qcrank_accuracy(expD,expMD,figId=1)

  
    if 'c' in args.showPlots:
        plot.dynamic_range(expD,expMD,figId=3)
    
    plot.display_all()
    print('M:done')
    #pprint(expMD)
