import numpy as np
import os, sys
from matplotlib import cm as cmap

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from Plotter_Backbone import Plotter_Backbone

#............................
#............................
#............................
class Circ_Plotter(Plotter_Backbone):
    def __init__(self, args):
        Plotter_Backbone.__init__(self,args)
        import qiskit
        print('qiskit ver=',qiskit.__qiskit_version__)
        for xx in [ self.outPath]:
            if os.path.exists(xx): continue
            print('Aborting on start, missing  dir:',xx)
            exit(99)


# figId=self.smart_append(figId)
#...!...!....................
    def save_me(self,fig,figId,tit=''): # added last plot to list
        self.plt.show()
        self.plt.tight_layout()
        figName='%s/%s_f%d.png'%(self.outPath,self.prjName,figId)
        print('Qiskit graphics saving to %s  ...'%figName)
        if len(tit)>0: fig.suptitle(tit)
        self.plt.savefig(figName)      

#...!...!....................
    def blank_page(self,figId,opt=None):
        figId=self.smart_append(figId)
        self.plt.figure(figId,facecolor='white', figsize=(6,3.7))
        nrow,ncol=1,1
        #  grid is (yN,xN) - y=0 is at the top,  so dumm
        ax = self.plt.subplot(nrow,ncol,1)
        #if 'title' in opt:
        #    if len(tit)>0: ax.set_title(opt['title'])
        return ax

#...!...!....................
    def circuit(self,circ,opt=None):
        from qiskit.tools.visualization import circuit_drawer , qx_color_scheme
        my_style={'usepiformat': True,
                  'showindex': True,
                  'cregbundle': True,
                  'compress': True,
                  'fold': 17,
                  'plotbarrier': True,
                  'output':'mpl' }
        fig=circuit_drawer(circ, style=my_style)
        return fig        

#...!...!....................
    def measurement(self,data,options={'figId':10,'sort': 'desc' }):
        figId=options['figId']
        figId=self.smart_append(figId)
        self.plt.figure(figId,facecolor='white', figsize=(6,3.7))
        nrow,ncol=1,1
        #  grid is (yN,xN) - y=0 is at the top,  so dumm
        ax = self.plt.subplot(nrow,ncol,1)

        baseV = sorted(data.keys())
        probV=[]; sigV=[];
        nShots=0
        for base in baseV:
            n,p,e=data[base]
            probV.append(p)
            sigV.append(e)
            nShots+=n
        
        width = 0.35  # the width of the bars
        error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
        
        errorL = [np.zeros(len(sigV)), sigV]
        ax.bar(baseV, probV, width=width, yerr=errorL, error_kw=error_kw, align='center', color='lightblue')

        tit='%s , shots=%d'%(self.prjName,nShots)
        ax.set(ylabel='probability',title=tit,xlabel='measurement basis')
        #ax.set_xticklabels(labels, fontsize=12, rotation=70)

        ymx=max(probV)+max(sigV)
        dy=ymx/30
        for x,y,s in zip(baseV, probV,sigV):
            ax.text(x, y+3*dy,'   %.3f' % y)
            ax.text(x, y+dy,'  (%.3f)' % s)
        
        ax.set_ylim(0., ymx+6*dy)
                   
        if 'sort' in options:
            if options['sort'] == 'asc':
                pass
            elif options['sort'] == 'desc':
                ax.invert_xaxis()
            else:
                bad44
