import numpy as np
import time,os,sys

from matplotlib import cm as cmap

sys.path.append(os.path.abspath("../../utils/"))
from Plotter_Backbone import Plotter_Backbone
from matplotlib.dates import DateFormatter

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

#............................
#............................
#............................
class OneExp_AnaPlotter(Plotter_Backbone):
    def __init__(self, args):
        Plotter_Backbone.__init__(self,args)
        self.probRange=args.probRange

            
#...!...!....................
    def histo_per_label(self,one,figId):
        nrow,ncol=1,8
        if len(one.measLab) >8:  nrow=2
        
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(15,2*nrow))        

        pMi,pMa=self.probRange
        binsX=np.linspace(pMi,pMa,40)
        #binsX=np.linspace(0.55, 0.85,40)
              
        #  grid is (yN,xN) - y=0 is at the top,  so dumm
        for idx,lab in enumerate(one.measLab):
            ax = self.plt.subplot(nrow,ncol,1+idx)
            ax.set(title='meas=%d'%(lab), xlabel='prob', ylabel='experiments')
            yA=one.Y[:,lab]
            ax.hist(yA,bins=binsX)
            ax.grid()

            avr,std,num,err=one.data_avr[lab]
            txt2='avr: %.3g\n +/- %.2g\nstd: %.2g'%(avr,err,std)
            if idx==0: txt2+='\nexp %s'%one.expId
            if idx==1: txt2+='\nsum %d'%num
            if idx==2: txt2+='\nrdErrCorr:%r'%one.conf['rdErrCorr']
            
            ax.text(0.15,0.3,txt2,color='b',transform=ax.transAxes)
            if lab==0: print(one.expId,lab,'=lab',txt2)
            

#...!...!....................
    def experiment_summary(self,one,figId):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(5,3))
        ax = self.plt.subplot()

        yB=one.Y  # [experiments][label]
        ax.boxplot(yB,labels=one.measLab,vert=False, showfliers=False) # last is no-outliers

        tit='exp:%s, %s'%(one.expId,str(one.jid6L[0]))
        if len(one.jid6L[0])>0: tit+=" x %d"%len(one.jid6L)

        ax.set(title=tit, xlabel='probability, shots=%d, cycles=%d'%(one.shots,one.numCyc) ,ylabel='measured bit-string')
        ax.grid()

        pMi,pMa=self.probRange
        ax.set_xlim(pMi,pMa)

        #print('eee',one.metaD.keys())
        txt1='exec_date '+one.metaD['retrInfo']['exec_date']
        txt1=txt1.replace(' ','\n')
        txt1=txt1.replace('+00:00','_UTC')
        x0=0.2
        ax.text(x0,0.6,txt1,transform=ax.transAxes,color='b')
        
        qL=one.conf['meas_qubit']
        txt2='meas Q:'+'-'.join([str(i) for i in qL])
        txt2+=' '+one.metaD['submInfo']['backName']
        txt2+='\nrdErrCorr=%r'%one.conf['rdErrCorr']
        ax.text(x0,0.4,txt2,transform=ax.transAxes,color='b')
