__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackbone import PlotterBackbone
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec

from matplotlib.colors import LinearSegmentedColormap

   
#...!...!....................
def summary_column(md):
    tmd=md['train']
    bmd=md['train']['best']
    cmd=md['circuit']
    ocf=md['opt_conf']
    
    txt='train job '+md['short_name']
    txt+='\ntest acc: %.3f'% bmd['test_acc']
    txt+='\nbest val acc:%.3f   steps:%d  fcnt:%d'%( bmd['val_acc'],bmd['steps'],bmd['fcnt'])
    txt+='\nbatchsize: %d'% tmd['batch_size']
    txt+='\nsamples %d'%tmd['tot_sampl']
    txt+='\nqubits: %d  params: %s'%(cmd['num_qubit'],cmd['param_shape'])
    txt+='\nanstaz: %s  layers:%d'%(ocf['ansatz_name'],ocf['ansatz_layers'])
    return txt
  
  
#............................
#............................
#............................
class Plotter(PlotterBackbone):
#...!...!....................
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

#...!...!....................
    def input_data(self,dom,md,bigD,figId=1,ax=None):
        X=bigD['X_'+dom]
        Y=bigD['Y_'+dom]
        if ax==None:
            nrow,ncol=1,1
            figId=self.smart_append(figId)
            fig=self.plt.figure(figId,facecolor='white', figsize=(6,5))
            ax = self.plt.subplot(nrow,ncol,1)

        ax.set_aspect(1.0) 
        ax.scatter(X[:, 0][Y == 1], X[:, 1][Y == 1], c="b", marker="o", ec="k")
        ax.scatter(X[:, 0][Y == -1], X[:, 1][Y == -1], c="r", marker="o", ec="k")
        inputN=md['data']['short_name']
        ax.set(title="input=%s, %s data "%(inputN,dom), xlabel='x0', ylabel='x1')

#...!...!....................
    def classified_data(self,dom,md,bigD,figId=1,ax=None):
        X=bigD['X_'+dom]
        Y=bigD['Y_'+dom]
        if ax==None:
            nrow,ncol=1,2
            figId=self.smart_append(figId)
            fig=self.plt.figure(figId,facecolor='white', figsize=(12,5.5))
            ax = self.plt.subplot(nrow,ncol,1)

        ax.set_aspect(1.0)
         
        inputN=md['data']['short_name']
        ax.set(title="classified=%s, %s data "%(inputN,dom), xlabel='x0', ylabel='x1')

        # plot data
        for color, label in zip(["b", "r"], [1, -1]):
            plot_x = X[:, 0][Y == label]
            plot_y = X[:, 1][Y == label]
            ax.scatter(plot_x, plot_y, c=color, marker="o", label="%s class=%d"%(dom,label))
        bmd=md['train']['best']
        lgTit='%s acc=%.3f   best: steps=%d  fcnt=%d'%(dom, bmd[dom+'_acc'],bmd['steps'],bmd['fcnt'])
        ax.legend(loc='upper center', title=lgTit,bbox_to_anchor=(0.5, 1.18), ncol=2)
        return ax
        
#...!...!....................
    def expval_contour(self,md,bigD,figId=1,ax=None):
        if ax==None:
            nrow,ncol=1,1
            figId=self.smart_append(figId)
            fig=self.plt.figure(figId,facecolor='white', figsize=(8,6.5))
            ax = self.plt.subplot(nrow,ncol,1)
        # plot decision regions
        xx=bigD['pred_contour_x0_bins']
        yy=bigD['pred_contour_x1_bins']
        Z=bigD['pred_contour_expval']

        levels = np.arange(-1, 1.1, 0.1)
        cm = self.plt.cm.RdBu       
        cnt = ax.contourf(xx, yy, Z, levels=levels, cmap=cm, alpha=0.8, extend="both")
        ax.contour(xx, yy, Z, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,))
        cbar=self.plt.colorbar(cnt, ticks=[-1, 0, 1])
        cbar.set_label('expectation value')  # Setting the label for the colorbar
        return ax
    
#...!...!....................
    def training_loss(self,task,figId=1):
        md=task.meta
        bigD=task.bigD
        nrow,ncol=1,2
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,4.5))
        ax = self.plt.subplot(nrow,ncol,1)

        stepV,trainAcc,valAcc,lrV=bigD['train_hist'].T
        print('sss',stepV.shape)
         
        # Plot y1 on the left axis
        line1 = ax.plot(stepV, trainAcc, 'b-', label='train acc')
        ax.plot(stepV, valAcc, 'r-', label='val acc')
        
        ax.set_ylabel('accuracy')
        ax.set_xlabel('optimizer steps')
        ax.grid()
        ax.axhline(0.5,linestyle='--',linewidth=1.0,color='k')

        bmd=md['train']['best']
        #print('zz',bmd['steps'],0, bmd['val_acc'],type(bmd['val_acc']))
        ax.axvline(bmd['steps'],linestyle='--',linewidth=1.0,color='r')
        ax.set_ylim(0.4,1.02)

        # . . . . . . . . . . . . . . . 
        # Plot  learning rate
        # Create a twin y-axis on the right side
        axr = ax.twinx()
        axr.spines['right'].set_visible(True)

        line2 = axr.plot(stepV,lrV , 'g--', label='learing rate')
        axr.set_ylabel('lr step size', color='g')
        axr.set_yscale('log')

        # Set the y-axis label colors to match the data colors
        #ax.tick_params(axis='y', labelcolor=line1[0].get_color())
        axr.tick_params(axis='y', labelcolor=line2[0].get_color())

        #ax1.set_xticks([0,1,2])
        #ax1.set_xticklabels(['0','1','2'])

        inputN=md['data']['short_name']
        elaT=md['train']['duration']/60.
        lgTit='job %s,  data: %s    train: %.1f min'%(md['short_name'],inputN,elaT)
        ax.legend(loc='upper center', title=lgTit,bbox_to_anchor=(0.5, 1.18), ncol=3)
        ax = self.plt.subplot(nrow,ncol,2)
        ax.axis('off')
        txt=summary_column(md)
        ax.text(0.02,0.45,txt,color='b', transform=ax.transAxes) #,va='bottom')
