#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Analyze   ProblemAtomGridSystem  experiment

- no graphics
'''

import os
from pprint import pprint
import numpy as np

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from ProblemAtomGridSystem import ProblemAtomGridSystem 
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

from toolbox.PlotterBackbone import PlotterBackbone
from matplotlib.colors import ListedColormap  # for herlading plot
import json
            
#I from quera_ahs_utils.plotting import show_register, show_drive_and_shift, show_global_drive
#I /usr/local/lib/python3.10/dist-packages/quera_ahs_utils/plotting.py

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3,4],  help="increase output verbosity", default=1, dest='verb')
         
    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or env--> $QuEra_dataVault")

    parser.add_argument('-e',"--expName",  default='exp_62a15b',help='AWS-QuEra experiment name assigned during submission')

    # plotting
    parser.add_argument("-p", "--showPlots",  default='ab', nargs='+',help="abc-string listing shown plots")
    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False, help="disable X-term for batch mode")

    args = parser.parse_args()
    # make arguments  more flexible
    if 'env'==args.basePath: args.basePath= os.environ['QuEra_dataVault']
    args.dataPath=os.path.join(args.basePath,'meas')
    args.outPath=os.path.join(args.basePath,'ana')
    args.showPlots=''.join(args.showPlots)
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.outPath)
    return args


#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

    #...!...!....................
    def show_register(self,md,figId=1,
                      blockade_radius: float=None,
                      what_to_draw: str="bond",
                      show_atom_index:bool=True):
        """Plot the given register extracted from meta-data

        Args:
            blockade_radius (float): None=use md,  The blockade radius for the register.
        what_to_draw (str): Default is "bond". Either "bond" or "circle" to indicate the blockade region. 
            show_atom_index (bool): Default is True. Choose if each atom's index is displayed over the atom itself in the resulting figure. 
           
        """

        figId=self.smart_append(figId)
        nrow,ncol=1,1

        fig=self.plt.figure(figId,facecolor='white', figsize=(6,5))
        ax = self.plt.subplot(nrow,ncol,1)
        
        um=1e-6 # conversion from m to um
        progD=json.loads( expD['program_org.JSON'][0])
        registerD=progD['setup']['ahs_register']

        #?registerD=md['payload']['program_org']['setup']['ahs_register']
        #pprint(registerD)

        # ......  Aquila FOV ......
        fov=md['payload']['aquila_area']
        fovX= float(fov['width'])/um
        fovY= float(fov['height'])/um
        nominal_Rb= md['payload']['nominal_Rb_um']

        ax.grid(ls=':')
        ax.set_aspect(1.)
        mm=2
        ax.set_xlim(-mm,fovX+10);        ax.set_ylim(-mm,fovY+10)        
        ax.add_patch( self.plt.Rectangle([0,0],fovX,fovY, color="g", alpha=0.3, fc="none", ls='--') )

        ax.plot([fovX,fovX+3],[fovY,fovY+3],color='g',ls='--')
        ax.text(fovX+3,fovY+3,'FOV',color='g',va='bottom')
        
        tit='GridArray job=%s'%(md['short_name'])
        ax.set(xlabel='X position (um)',ylabel='Y position (um)',title=tit)

        #.... misc text
        txt1='back: '+md['submit']['backend']
        txt1+=', atoms: %d'%md['payload']['num_atom']
        txt1+=', omega: %.2f MHz'%md['payload']['rabi_omega_MHz']
        txt1+=', Rb: %.1f um'%nominal_Rb
        ax.text(0.0,fovY+7,txt1,color='g')
        
        # ....  Atoms  ............
        atoms=[ ]; voids=[]
        for [cx,cy],f in zip( registerD['sites'], registerD['filling']):
            xy=[float(cx)/um,float(cy)/um]
            if f:
                atoms.append(xy )
            else:
                voids.append( xy)
                
        atoms=np.array(atoms)
        voids=np.array(voids)
        #print('atoms:',atoms,atoms.size)

        if atoms.size>0: # filled_sites
            ax.plot(atoms[:,0], atoms[:,1], 'r.', ms=15, label='filled')
        if voids.size>0: # empty_sites
            ax.plot(voids[:,0], voids[:,1],  'k.', ms=5, label='empty')

        
        nA=atoms.shape[0]
    
        if what_to_draw=="circle":
            for i in range(nA):
                x,y=atoms[i]
                #  Rb radii
                ax.add_patch( self.plt.Circle((x,y), nominal_Rb/2, color="b", alpha=0.3) )
                ax.add_patch( self.plt.Circle((x,y), nominal_Rb, color="b", alpha=0.1) )
                ax.add_patch( self.plt.Circle((x,y), nominal_Rb*2, color="b", alpha=0.03) )
                # add label atoms
                ax.text(x+0.5,y+1,'%d'%i,color='k',va='center')

        if what_to_draw=="bond":            
            for i in range(nA):
                for j in range(i+1, nA):
                    A=atoms[i]; B=atoms[j]
                    dist = np.linalg.norm(A-B)
                    if dist > nominal_Rb: continue
                    ax.plot([A[0],B[0]], [A[1],B[1]], 'b')
                    
        return atoms, ax
    
    #...!...!....................
    def counts(self,ax,atoms,md,expD):
        counts=expD['counts_atom']
        shots=md['submit']['num_shots']
        nAtom=md['payload']['num_atom']
        fov=md['payload']['aquila_area']
        um=1e-6 # conversion from m to um
        fovX= float(fov['width'])/um
        fovY= float(fov['height'])/um
        print('cnt',counts)
        
        for j in range(nAtom):
            ng,nr=counts[j]
            #print(j,ng,nr)
            ne=shots-ng-nr
            txt='%d:%d'%(ng,nr)             
            if ne>0: txt+=' e%d'%ne
            x,y=atoms[j]
            #print(j,x,y,txt)
            ax.text(x+0.5,y+2,txt,color='b',va='center')
        txt1='shots: %d'%shots
        txt1+=', exec:'+md['job_qa']['exec_date']
        ax.text(0.02,fovY+3.5,txt1,color='b')

        ax.text(fovX,fovY-8,'counts\n g:r (e)',color='b')

    #...!...!....................
    def heralding_per_atom(self,md,expD,figId=1,                           ):
        """Plot atoms population at T0 
        INPUT:
        - geometry from MD
        - heralding stats from .expD['herald_prob_atom']
        """

        figId=self.smart_append(figId)
        nrow,ncol=1,1

        fig=self.plt.figure(figId,facecolor='white', figsize=(6,5))
        ax = self.plt.subplot(nrow,ncol,1)
       
        um=1e-6 # conversion from m to um
        progD=json.loads( expD['program_org.JSON'][0])
        registerD=progD['setup']['ahs_register']
        #pprint(registerD)
      
        # compute average occupancy probability
        probs=expD['herald_prob_atom']
        print('probs:',probs.shape)
        #print('cc1',probs)
        avrProb=np.mean(probs[0,:,1])
        avrShots=np.mean(probs[2,:,1])
        print('avrProb of placing atom: %.3f, avrShots=%.1f'%(avrProb,avrShots))

        #probs[0,5,1]=0.4
 
        # ......  Aquila FOV ......
        fov=md['payload']['aquila_area']
        fovX= float(fov['width'])/um
        fovY= float(fov['height'])/um
        nominal_Rb= md['payload']['nominal_Rb']

        ax.grid(ls=':')
        ax.set_aspect(1.)
        mm=2
        ax.set_xlim(-mm,fovX+10);        ax.set_ylim(-mm,fovY+10)        
        ax.add_patch( self.plt.Rectangle([0,0],fovX,fovY, color="g", alpha=0.3, fc="none", ls='--') )

        ax.plot([fovX,fovX+3],[fovY,fovY+3],color='g',ls='--')
        ax.text(fovX+3,fovY+3,'FOV',color='g',va='bottom')
        
        tit='Heralding for job=%s'%(md['short_name'])
        ax.set(xlabel='X position (um)',ylabel='Y position (um)',title=tit)

        #.... misc text
        txt1='back: '+md['submit']['backend']
        txt1+=', atoms: %d'%md['payload']['num_atom']
        txt1+=', avr fill %.3f'%avrProb
        txt1+=', avr shots %.1f'%avrShots
        ax.text(0.0,fovY+7,txt1,color='g')

        # color map
        my_colors = [ '#006837', '#1a9641',  '#a6d96a','#c0c0c0','#c0c0c0','yellow','#fdae61','#d7191c']
        numCol=len(my_colors)
        my_cmap = ListedColormap(my_colors) # Create a custom colormap       
        zticks= [i for i in range(-4, 4+1)]
        # Add colorbar representing the circle index based on the sine of the index
        cbar = self.plt.colorbar(self.plt.cm.ScalarMappable(cmap=my_cmap), ax=ax)
        cbar.set_label('fill/avr_fill (std dev)')
        
        #zticks = [-4,-3,-2,-1,0,1,2,3,4]
        cbar.set_ticks(np.linspace(0, 1, num=len(zticks)))
        cbar.set_ticklabels(zticks)
        
        # ....  Atoms  ............
        atoms=[ ]
        j=0
        for [cx,cy],f in zip( registerD['sites'], registerD['filling']):
            pr,erPr=probs[:2,j,1]
            nSig=(pr - avrProb)/erPr
            nSig=np.clip(nSig,-3.99,3.99)
            iCol= int(nSig+4)
            #print('j=%d pr %.2f +/- %.2f, nSig=%.1f, icol=%d'%(j,pr,erPr,nSig,iCol))
            col=my_colors[iCol]
            xyc=[float(cx)/um,float(cy)/um,col,nSig]
            atoms.append(xyc )                        
            j+=1  # indexes atoms
                
        nA=len(atoms)
        #print(nA,'atoms:',atoms)

        for i in range(nA):
            x,y,c,s=atoms[i]
            ax.plot(x,y, '.', ms=25,color=c)
            ax.text(x,y,'%.1f'%s)
        


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=2)
                    
    inpF=args.expName+'.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))

    if 1: # patch June6 data
        #from decimal import Decimal
        #expMD['payload']['evol_time_ramp_us']=Decimal('0.1') 
        #expMD['payload']['nominal_Rb_um']=expMD['payload']['nominal_Rb']
        a=1
        
    if args.verb>=2:
        print('\nM:expMD:');  pprint(expMD)
        if args.verb>=3:
            keyL=list(expD)
            for x in  keyL:
                if 'JSON' in x: 
                    jrec=expD.pop(x)[0]
                    x1=x.replace('.JSON','')
                    expD[x1]=json.loads(jrec)
                    print('\nDataJ:',x1,expD[x1])
                else:
                    print('\nData:',x,expD[x])
        stop3

        
    task= ProblemAtomGridSystem(args,expMD,expD)

    task.analyzeExperiment()
    
    #...... WRITE  OUTPUT
    outF=os.path.join(args.outPath,expMD['short_name']+'.ags.h5')
    write4_data_hdf5(task.expD,outF,expMD)

    # ----  just plotting
    args.prjName=expMD['short_name']
    plot=Plotter(args)
    
    if 'a' in args.showPlots:
        plot.show_register(expMD, what_to_draw="circle",figId=1)
    if 'b' in args.showPlots:
        atoms_um,ax=plot.show_register(expMD, what_to_draw="bond",figId=2)
        plot.counts(ax,atoms_um,expMD,expD)
    if 'h' in args.showPlots:
        plot.heralding_per_atom(expMD,expD,figId=3)
    plot.display_all(png=1)

    print('M:done')


