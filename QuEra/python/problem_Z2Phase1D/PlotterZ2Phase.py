__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
from pprint import pprint
from bitstring import BitArray
import networkx as nx  # for graph generation

from toolbox.PlotterQuEra import PlotterQuEra

#............................
#............................
#............................
class Plotter(PlotterQuEra):
    def __init__(self, args,prjName=None):
        if prjName!=None: args.prjName=prjName
        PlotterQuEra.__init__(self,args)
   

    #...!...!....................
    def correlations(self,task,corType,figId=3):
        figId=self.smart_append(figId)
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,6))
        md=task.meta
        smd=md['submit']
        amd=md['postproc']
        pd=md['payload']
        nAtom= amd['num_atom']
        nominal_Rb= pd['nominal_Rb_um']

        #... get data
        if corType=='density':
            gij=task.expD['dens_corr']
            deRyd=task.expD['dens_rydberg']
            vmax=0.25; vmin=-vmax
        if corType=='magnet':
            gij=task.expD['magnet_corr']
            twoPtCorr=task.expD['magnet_2pt_corr']
            vmax=1; vmin=-1
        #::::::::::::::::  2D density  ::::::::::::::::  
        ax = self.plt.subplot(nrow,ncol,1)
        ax.set_aspect(1.)
        
        image = ax.imshow(gij, cmap='bwr', vmin=vmin, vmax=vmax)
        self.plt.colorbar(image, ax=ax)
        tit=corType+' corr, %d atoms chain,  job=%s'%(nAtom,md['short_name'])
        ax.set(xlabel="atom index",ylabel="atom index",title=tit)

        atomTicks=[j for j in range(0,nAtom,2)]
        ax.set_xticks(atomTicks)
        ax.set_yticks(atomTicks)
        
        #.... misc text
        txt1='back: '+md['submit']['backend']
        txt1+=', atoms: %d'%md['payload']['num_atom']
        txt1+=', omega: %.1f MHz'%md['payload']['rabi_omega_MHz']
        txt1+=', Rb: %.1f um'%nominal_Rb
        ax.text(0,0,txt1,color='g')
        #ax.text(0.02,0.92,txt1,color='g', transform=ax.transAxes)
        
        if corType=='density':   # special case for 1D density
            ax = self.plt.subplot(nrow,ncol,2)
            binX=[i-0.5 for i in range(nAtom+1) ]
            ax.stairs(deRyd,binX, fill=True)
            tit='avr Rydberg dens, %d atoms chain,  job=%s'%(nAtom,md['short_name'])
            ax.set(xlabel="atom index",ylabel="probability",title=tit)
            ax.grid()
            ax.set_ylim(0,1.02)
            return

        #::::::::::::::::  two-poin correlation  ::::::::::::::::  
        ax = self.plt.subplot(nrow,ncol,2)
        ax.plot(twoPtCorr[:,0], twoPtCorr[:,1],'*')
        tit=corType+' 2-point corr, %d atoms chain,  job=%s'%(nAtom,md['short_name'])
        ax.set(xlabel="displacement (sites)",ylabel=r"two-point correlation $\zeta$",title=tit)
        ax.grid()
        ax.set_yscale('log')
        
        #... get fit results
        XYf=task.expD['2pt_corr_fit']
        Xf = XYf[0]
        Yf = XYf[1]
        ax.plot(Xf,Yf,'-')
        # Embed the fit parameters in the image
        fmd=task.meta['2pt_corr_fit']
        text='fit:  %.2f * exp ( -L / %.2f )'%(fmd['a'],-1/fmd['b'])
        ax.text(0.4, 0.9, text, transform=ax.transAxes,c='b', fontsize=14)

       
#...!...!....................
    def Z2Phase_solutions(self,task,nSol,figId=3):
        figId=self.smart_append(figId)
        nrow,ncol=2,round(0.1+nSol/2)
        if nSol==3: nrow,ncol=1,3

        fig=self.plt.figure(figId,facecolor='white', figsize=(4*ncol,4.5*nrow))
        md=task.meta
        smd=md['submit']
        amd=md['postproc']
        pd=md['payload']
        nAtom= amd['num_atom']
        um=1e-6 # conversion from m to um
        
        # ......  Aquila FOV ......
        fov=md['payload']['aquila_conf']['area']
        fovX= float(fov['width'])/um
        fovY= float(fov['height'])/um
        nominal_Rb= md['payload']['nominal_Rb_um']
        
        G=task.graph
        m=1  #tmp  for loop+anc - there is some bug
        m=0 # regular chain 
        pos=task.atoms_pos_um()[:nAtom-m] +(0,3)  # move it vertically a bit
        # Print the graph
        #print('nodes:',G.nodes); print(G.edges)
        #print('pos_um',pos)
        
        probV = task.expD['ranked_probs']  # {prob,probEr,mshot} x {pattern index}
        hexpattV= task.expD['ranked_hexpatt']
        hammwV=task.expD['ranked_hammw']
        energyV=task.expD['ranked_energy_eV']
        
        nSigThr=1.  # cuto-off for signifficance of results
        for k in range(nSol):
            ax = self.plt.subplot(nrow,ncol,1+k)
            ax.set_aspect(1.)
            hexpatt=hexpattV[k]
            #print('aaa',hexpatt,nAtom,type(hexpattV)) 
            A=BitArray(hex=hexpatt)[-nAtom:]  # clip leading 0s
            prob,probEr,mshot=probV[:,k]
            card=hammwV[k]
            nSig=prob/probEr
            print('patt=',hexpatt,prob,probEr,mshot,'nSig=%.1f'%nSig)
            #1if nSig<nSigThr: continue
            vert_colors = ['tomato' if (bitc == '1') else 'aqua' for bitc in A.bin ]
            if m: vert_colors = vert_colors[:-1]
            
            #print('ddd',len(G), len(pos),len(vert_colors))
            doNodeLab= self.venue=='prod'
            nx.draw(G, pos = pos, ax=ax, with_labels=doNodeLab, node_color=vert_colors)
            if 'YrangeLR' in md:
                ax.set_ylim(tuple(md['YrangeLR']))
                doFOV=False
            else:
                ax.set_ylim(-5,fovX+10);

            if self.venue=='paper':
                dLab=chr(97 + k) + ')'
                ax.text(0.03,0.99,dLab,color='k', transform=ax.transAxes, size=20)
                continue
            tit='hex:%s  prob: %.3f+/-%.3f'%(hexpatt,prob,probEr)
            tit+='\ni:%d  card:%d  ene:%.3g (eV)\nmshot:%d/%d'%(k,card,energyV[k],mshot,amd['used_shots'])
            #print('sol:',tit)
            ax.set(title=tit) 
            
            txt1=None
            if k==0: txt1=md[ 'payload']['info'].replace(",","\n")
            if k==1: txt1=md['submit']['info'].replace(",","\n")
            if k==2: txt1=md['job_qa']['info'].replace(",","\n")
            if txt1!=None: ax.text(0.02,0.80,txt1,color='g', transform=ax.transAxes,va='top')

        return ax  # will be the last plot
