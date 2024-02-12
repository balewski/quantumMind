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

#...!...!....................
    def one_label_vs_date(self,one,targetLab,obsN,figId):        
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,3))
        ax = self.plt.subplot()

        assert obsN=='counts'
        assert len(one.execT)>0
        iLab=one.labels.index(targetLab)
        print(targetLab,iLab)

        Y=one.Y        
        #print('P: Y sh',Y.shape)
        
        tmV=one.execT
        yV=[]
        yV=Y[:,:,iLab]
        #print('ttt',tmV)
        #print('yyy',yV.shape)
        tm0=min(tmV)
        #print('minT',tm0)
        dtmV=[ (x-tm0).total_seconds()/3600. for x in tmV ]
        #print('ddd',dtmV)
        #print('jid',one.jid6L)
        
        
        ax.plot(dtmV , yV,  'o')
        # do box-plot instead
        #1yB=[ yV[i,:] for i in range(yV.shape[0]) ]
        #1dtmV=[ float('%.1f'%x) for x in dtmV ]
        #1ax.boxplot(yB,labels=dtmV, showfliers=False) # last is no-outliers

        tit='meas=%s, num jobs=%d, shots=%d'%(targetLab,Y.shape[0],one.shots)
        ax.set(title=tit, xlabel='wall hours since: %s'%tm0, ylabel='%s'%obsN)
        ax.grid()

        for a,b,t in zip(dtmV,yV[:,0],one.jid6L): #
            ax.text(a,b,t,rotation=60)

            
#...!...!....................
    def histo_per_label(self,one,obsN,figId):
        nrow,ncol=1,8
        if len(one.labels) >8:  nrow=2
        
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(15,2*nrow))        
        if obsN=='prob':
            pmax=1.02
            #binsX=np.linspace(0., pmax,40)
            binsX=np.linspace(0.55, 0.85,40)
        else:
            binsX=40
              
        #  grid is (yN,xN) - y=0 is at the top,  so dumm
        for idx,lab in enumerate(one.labels):
            ax = self.plt.subplot(nrow,ncol,1+idx)
            ax.set(title='meas=%s'%(lab), xlabel=obsN, ylabel='experiments')
            yA=one.dataV[obsN][idx]
            ax.hist(yA.flatten(),bins=binsX)
            ax.grid()

            avr,std,num,err=one.dataV['avr_'+obsN][idx]
            txt2='avr: %.3g\n  +/-%.2g'%(avr,err)
            if idx==0: txt2+='\nexp %s'%one.expId
            if idx==1: txt2+='\nsum %d'%num
            ax.text(0.15,0.5,txt2,color='b',transform=ax.transAxes)

#...!...!....................
    def experiment_summary(self,one,obsN,figId):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(5,3))
        ax = self.plt.subplot()

        yA=one.dataV[obsN] # [label][experiments]
        #print('P: ex0', yA.shape)
        [numLab,numExp,numCyc]=yA.shape

        yB=[ yA[i,:].flatten() for i in range(yA.shape[0]) ]
        
        ax.boxplot(yB,labels=one.labels,vert=False, showfliers=False) # last is no-outliers

        tit=one.metaD['submInfo']['transpName']+', exp:%s, %s'%(one.expId,str(one.jid6L[0]))
        if len(one.jid6L[0])>0: tit+=" x %d"%len(one.jid6L)

        ax.set(title=tit, xlabel=obsN+', shots=%d, cycles=%d'%(one.shots,numCyc) ,ylabel='measured bit-string')
        ax.grid()

        if obsN=='prob':
            pmax=0.9
        ax.set_xlim(0,pmax)

        #print('eee',one.metaD.keys())
        txt1='exec_date '+one.metaD['retrInfo']['exec_date']
        txt1=txt1.replace(' ','\n')
        txt1=txt1.replace('+00:00','_UTC')
        ax.text(0.1,0.4,txt1,transform=ax.transAxes,color='b')

#...!...!....................
    def multi_experiment(self,oneD,lab,expIdL,obsN,figId):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,3))

        print('muti_experiment, label=',lab)
        oneBase=oneD['0.base.0']
        txt1=oneBase.metaD['submInfo']['transpName']+',   jid: %s'%(str(oneBase.jid6L))
        #print('txt1=',txt1)
        tl=120
        if len(txt1) >tl : txt1=txt1[:tl]+'clip'

        iLab=None
        pV=[]
        for expId in expIdL:
            one=oneD[expId]
            if iLab==None:
                iLab=one.labels.index(lab)
                #print('P2:tl',lab,iLab)

            yA=one.dataV[obsN][iLab] # [label][experiments]
            #print('P: ex2', yA.shape)
            (avr,std,num,err)=yA
            pV.append([avr,err])
        pV=np.array(pV)
        #print('P2:pV',pV.shape)

        ax = self.plt.subplot()
        xidx=range(len(expIdL))
        ax.errorbar(xidx ,  pV[:,0], yerr=pV[:,1], fmt='o', ecolor='g')
        yy=pV[0,0]
        ax.axhline(yy, linestyle='--', linewidth=1)

        y00=0.2
        if obsN=='avr_prob': y00=0.8
        ax.text(0.2,y00,txt1,color='b',transform=ax.transAxes)

        ax.set(title=obsN+' (meas=%s), noise injected at 1 gate'%lab,  xlabel='noise injection gate index:  circ_layer.gate_name.qubit(s)', ylabel=obsN)
        ax.grid()
        ax.set_xticks(xidx)
        ax.set_xticklabels(expIdL,rotation=80)
    


#...!...!....................
    def histo_per_labelXX(self,one,obsN,figId):
        nrow,ncol=1,2
         
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(7,2*nrow))        
        if obsN=='prob':
            pmax=1.02
            binsX=np.linspace(0., pmax,40)
            
        else:
            binsX=40
              
        #  grid is (yN,xN) - y=0 is at the top,  so dumm
        for idx,lab in enumerate(one.labels):
            print('zzz',idx,lab)
            if idx==0: binsX=np.linspace(0.70, 0.85,100)
            else:   binsX=np.linspace(0.0, 0.15,100)
            ax = self.plt.subplot(nrow,ncol,1+idx)
            ax.set(title='meas=%s'%(lab), xlabel=obsN, ylabel='experiments')
            yA=one.dataV[obsN][idx]
            ax.hist(yA.flatten(),bins=binsX)
            ax.grid()

            avr,std,num,err=one.dataV['avr_'+obsN][idx]
            txt2='avr: %.4g\n  +/-%.2g'%(avr,err)
            if idx==0: txt2+='\n'+one.metaD['submInfo']['transpName']
            if idx==1: txt2+='\nsum %d'%num
            ax.text(0.1,0.5,txt2,color='b',transform=ax.transAxes)
            if idx>0: break
