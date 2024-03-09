#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 generate data samples of various types to be used by QML

'''

import os
from pprint import pprint
import numpy as np
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.PlotterBackbone import PlotterBackbone

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or 'env'")
    parser.add_argument("-s","--select",  default=['circle','2'],  nargs='+',help="type of data set, tuple(tag, par1,par2...), see buildMeta(.) ")
    parser.add_argument('-i','--numSample', default=1500, type=int, help='num of images packed in to the job')
    parser.add_argument('-d',"--dataName",  default=None,help='optional name of the data set')
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")
    parser.add_argument("-p", "--showPlots",  default='a  ', nargs='+',help="abc-string listing shown plots")
     
    
    args = parser.parse_args()
    # make arguments  more flexible
    if 'env'==args.basePath: args.basePath= os.environ['PennyLane_dataVault']
    args.dataPath=os.path.join(args.basePath,'input')
    args.outPath='out/'
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.dataPath)
    
    return args

#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

    #...!...!....................
    def circle(self,md,bigD,figId=1,ax=None,tit=None):
        X=bigD['data_X']
        Y=bigD['data_Y']
        if ax==None:
            nrow,ncol=1,1
            figId=self.smart_append(figId)
            fig=self.plt.figure(figId,facecolor='white', figsize=(6,5))
            ax = self.plt.subplot(nrow,ncol,1)

        ax.set_aspect(1.0) 
        ax.scatter(X[:, 0][Y == 1], X[:, 1][Y == 1], c="b", marker="o", ec="k")
        ax.scatter(X[:, 0][Y == -1], X[:, 1][Y == -1], c="r", marker="o", ec="k")
        ax.set(title="Original data", xlabel='x0', ylabel='x1')


#............................
#........E.N.D...............
#............................

#...!...!....................
def generate_circle(md,bigD):
    iDim=md['inp_dim']
    nSamp=md['num_sample']
    #sh=(nSamp,iDim)
    X= np.random.uniform(-1,1, size=( nSamp,iDim))
    x0=[0., -0.] ; Rcut=0.8  # for equal balance
    #x0=[0.6, -0.3] ; Rcut=0.9

    X1=X - x0
    # divide data into 2 classes 
    l2 = np.linalg.norm(X1, axis=1)
    #print('L2',l2)
    # Generate labels based on the distances
    Y = np.where(l2 <  Rcut, 1, -1)
    ny1=sum(Y>0); print('Y1=%d, prob=%.2f'%(ny1,ny1/nSamp))
    print('Y:',Y[:20])
    md['balance']=ny1/nSamp
    bigD['data_X']=X
    bigD['data_Y']=Y
    
#...!...!....................
def generate_tort(md,bigD):
    iDim=md['inp_dim']
    nSamp=md['num_sample']
    sh=(nSamp,iDim)
    X= np.random.uniform(-1,1, size=( nSamp,iDim))
    M=md['num_slice']
    # Calculate angles in radians
    angles = np.arctan2(X[:, 1], X[:, 0])

    # Normalize angles to be in [0, 2*pi)
    angles = np.mod(angles, 2*np.pi)

    # Determine the sector index for each point
    sector_indices = np.floor(M * angles / (2*np.pi)).astype(int)

    # Label sectors: +1 for even-indexed sectors, -1 for odd-indexed sectors
    Y = np.where(sector_indices % 2 == 0, 1, -1)

    ny1=sum(Y>0); print('Y1=%d, prob=%.2f'%(ny1,ny1/nSamp))
    print('Y:',Y[:20])
    md['balance']=ny1/nSamp
    bigD['data_X']=X
    bigD['data_Y']=Y
    

#...!...!....................
def buildMeta(args):
    selL=args.select
    selN=selL[0]
    md={}
    if selN=='circle':
        iDim=int(selL[1])  # input dimension
        oCat=2  # output categories
        md['type']='circle'
        md['inp_dim']=iDim
        md['num_class']=oCat # for now
        md['num_sample']=args.numSample
        dataN='%s%dd%dc'%(selN,iDim,oCat)
        md['comment']='non-linearly separable, categorical'

    if selN=='tort':
        iDim=int(selL[1])  # input dimension
        nSlice=int(selL[2])
        assert nSlice%2==0
        oCat=2  # output categories
        md['type']='tort'
        md['inp_dim']=iDim
        md['num_class']=oCat # for now
        md['num_slice']=nSlice
        md['num_sample']=args.numSample
        dataN='%s%dd%dcM%d'%(selN,iDim,oCat,nSlice)
        md['comment']='non-linearly separable, categorical'

    if args.dataName!=None: dataN=args.dataName
    md['short_name']=dataN
    return md

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()

    outMD=buildMeta(args)
    pprint(outMD)
    bigD={}
    if outMD['type']=='circle':
       generate_circle(outMD,bigD)

    if outMD['type']=='tort':
       generate_tort(outMD,bigD)

    if args.verb>0: pprint(outMD)
    #...... WRITE  OUTPUT .........

    outF=os.path.join(args.dataPath,outMD['short_name']+'.h5')
    
    write4_data_hdf5(bigD,outF,outMD)

    # ----  just plotting
    args.prjName=outMD['short_name']
    plot=Plotter(args)
    if 'a' in args.showPlots:
        plot.circle(outMD,bigD,figId=1)

    plot.display_all(png=1)



