import numpy as np
import time,os,sys

from matplotlib import cm as cmap

sys.path.append(os.path.abspath("../../utils/"))
from Plotter_Backbone import Plotter_Backbone

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

#............................
#............................
#............................
class Plotter_HWcalib(Plotter_Backbone):
    def __init__(self, args):
        Plotter_Backbone.__init__(self,args)

    #...!...!....................
    def qubits_calib(self,valD, metaDA,figId):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(9,5))

        nrow,ncol=2,4
        metaD= metaDA['qubits']
        valNameL=sorted(list(valD.keys()))
        xlab=['q%d'%x for x in metaD['Qid']]
        xidx=range(len(xlab))
        
        for idx,name in enumerate(valNameL):
            j=idx
            if idx>=3: j=idx+1 # ugly, must match with code below
            
            #  grid is (yN,xN) - y=0 is at the top,  so dumm
            ax = self.plt.subplot(nrow,ncol,1+j)
            valV=valD[name]
            ax.plot(xidx, valV ,  'o')
            ax.set(title='%s'%name, xlabel='qubit ID', ylabel='%s'%metaD[name]['unit'])
            ax.grid()
            ax.set_xticks(xidx)
            ax.set_xticklabels(xlab,rotation=60)
                         
        
        # histo of time-lag
        titTL=['qubits calib','gates calib']
        valTL=[metaDA['qubits']['timeLag'],
               metaDA['gates1Q']['timeLag'] + metaDA['gates2Q']['timeLag']]

        txt1='calib_date '+metaDA['calib_date']
        txt1=txt1.replace(' ','\n')

        for i in range(2):
            ax = self.plt.subplot(nrow,ncol,4+i*4)
            valV=np.array(valTL[i])/3600.
            
            binsX=50
            ax.hist(valV,bins=binsX)
            ax.set(title=titTL[i], xlabel='time lag (h)', ylabel='observables')
            if i==0:
                ax.text(0.1,0.4,txt1,transform=ax.transAxes,color='b')
            if i==1:
                ax.text(0.1,0.4,metaDA['backend'],transform=ax.transAxes,color='b')

        
    #...!...!....................
    def gates1Q_calib(self,valD, metaD,figId):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,5))

        nrow,ncol=2,4
        metaD= metaD['gates1Q']
        valNameL=sorted(list(valD.keys()))
        
        xlab=['q%d'%x for x in metaD['Qid']]
        xidx=range(len(xlab))
        obsL=metaD['observable']
        assert len(obsL)==2
        scaleA=np.array([100,1])
        ylab=metaD['unit']
        ylab[0]='error in %c'%37
        for idx,name in enumerate(valNameL):
            valV=np.array(valD[name])*scaleA
            
            #  grid is (yN,xN) - y=0 is at the top,  so dumm
            for j in range(2):
                ax = self.plt.subplot(nrow,ncol,1+idx+j*ncol)
                ax.plot(xidx, valV[:,j] ,  'o') 
                ax.set(title='%s %s'%(name,obsL[j]), xlabel='qubit ID', ylabel='%s'%ylab[j])
                ax.grid()
                ax.set_xticks(xidx)
                ax.set_xticklabels(xlab,rotation=60)

        axE = self.plt.subplot(nrow,ncol,2+idx)
        axT = self.plt.subplot(nrow,ncol,2+idx+1*ncol)
        return [axE,axT]

    #...!...!....................
    def gates2Q_calib(self,valD, metaD,axAr):
        name='cx'
        metaD= metaD['gates2Q']
        xlab= metaD['Qid2']
        xidx=range(len(xlab))
        obsL=metaD['observable']
        assert len(obsL)==2
        scaleA=np.array([100,1])
        ylab=metaD['unit']
        ylab[0]='error in %c'%37
        
        valV=np.array(valD[name])*scaleA
        #  grid is (yN,xN) - y=0 is at the top,  so dumm
        for j in range(2):
            ax = axAr[j]
            ax.plot(xidx, valV[:,j] ,  'o') 
            ax.set(title='%s %s'%(name,obsL[j]), xlabel='qubit pairs', ylabel='%s'%ylab[j])
            ax.grid()
            ax.set_xticks(xidx)
            ax.set_xticklabels(xlab,rotation=75)

   
