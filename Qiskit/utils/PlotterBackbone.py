#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os

#............................
#............................
#............................
class PlotterBackbone(object):
    def __init__(self, args):
        self.jobName=args.prjName
        # important : do not move this import outside the scope of this class
        import matplotlib as mpl
        if args.noXterm:
            if args.verb>0: print('disable Xterm')
            mpl.use('Agg')  # to plot w/o X-server
        else:
            mpl.use('TkAgg')
        import matplotlib.pyplot as plt

        if args.verb>0: print(self.__class__.__name__,':','Graphics started')
        plt.close('all')
        self.plt=plt
        self.args=args
        self.figL=[]
        self.outPath=args.outPath+'/'
        assert os.path.exists(self.outPath)

    #............................
    def figId2name(self, fid):
        figName='%s_f%d'%(self.jobName,fid)
        return figName

    #............................
    def clear(self):
        self.figL=[]
        self.plt.close('all')
        
    #............................
    def canvas4figure(self,figPayload,figId=10):
        # input:  <class 'matplotlib.figure.Figure'> w/o Canvas
        # this code adds a canvas manager to an existing figure, e.g. from Qiskit:StandardRB
        
        figId=self.smart_append(figId)
        dummy=self.plt.figure(figId, figsize=(6,4))
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure =figPayload 
        figPayload.set_canvas(new_manager.canvas)

    #............................
    def display_all(self, png=1):
        if len(self.figL)<=0: 
            print('display_all - nothing top plot, quit')
            return
        
        for fid in self.figL:
            self.plt.figure(fid)
            self.plt.tight_layout()
            figName=os.path.join(self.outPath,self.figId2name(fid))
            if png: figName+='.png'
            else: figName+='.pdf'
            print('Graphics saving to ',figName)
            self.plt.savefig(figName)
        self.plt.show()

        
# figId=self.smart_append(figId)
#...!...!....................
    def smart_append(self,id): # increment id if re-used
        while id in self.figL: id+=1
        self.figL.append(id)
        return id

#............................
#............................
#............................
# example class using the plotter 
class PlotterEnergyUse(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

    #...!...!....................
    def pageA(self,metaD,bigD,figId=5):
        nrow,ncol=1,1
        #  grid is (yN,xN) - y=0 is at the top,  so dumm
        figId=self.smart_append(figId)
        self.plt.figure(figId,facecolor='white', figsize=(6,4))
        ax=self.plt.subplot(nrow,ncol,1)
        tit='job=%s'%(metaD['name1'])
        xV=bigD['xV']
        fV=bigD['f1']
        ax.plot(xV,fV)
        ax.set(xlabel='wall time (sec)',ylabel='power (W)', title=tit)


    #...!...!....................
    def pageB(self,metaD,bigD,figId=5):
        nrow,ncol=1,2
        #  grid is (yN,xN) - y=0 is at the top,  so dumm
        figId=self.smart_append(figId)
        self.plt.figure(figId,facecolor='white', figsize=(5,3))
        ax=self.plt.subplot(nrow,ncol,1)
        tit='job=%s'%(metaD['name1'])
        xV=bigD['xV']
        fV=bigD['f2']
        ax.plot(xV,fV)

        
#=================================
#=================================
#  M A I N
#=================================
#=================================

import numpy as np

if __name__ == "__main__":
    
    class EmptyTrunk:    pass  # empty class  
    args=EmptyTrunk() # it should be argparse
    args.outPath='./' #  plots area saved here
    args.prjName='abc' # prefix to all plots
    args.verb=1 # verbosity level
    args.noXterm=nFalse # set it to true to suppres canvas popup , if you have no X11 forwarding
    

    # prepare data+meta-data
    metaD={'name1':'my func'}
    xV=np.arange(0., 40)* np.pi/100.
    f1=np.sin(xV)
    f2=np.tan(xV)

    bigD={'xV':xV,'f1':f1,'f2':f2}
    
    # ----  just plotting
    plot=PlotterEnergyUse(args)

    plot.pageA(metaD,bigD)
    plot.pageB(metaD,bigD)

    plot.display_all()  # now all plots are created
