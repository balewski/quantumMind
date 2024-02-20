__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import json

from toolbox.PlotterBackbone import PlotterBackbone
#............................
#............................
#............................
class PlotterQuEra(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

    #...!...!....................
    def show_register(self,task,figId=1,
                      blockade_radius: float=None,
                      what_to_draw: str="bond",
                      show_atom_index:bool=True,
                      ax=None):
        """Plot the given register extracted from meta-data

        Args:
            blockade_radius (float): None=use md,  The blockade radius for the register.
        what_to_draw (str): Default is "bond". Either "bond" or "circle" to indicate the blockade region. 
            show_atom_index (bool): Default is True. Choose if each atom's index is displayed over the atom itself in the resulting figure. 
           
        """

        if ax==None:
            figId=self.smart_append(figId)
            nrow,ncol=1,1
            fig=self.plt.figure(figId,facecolor='white', figsize=(6,5))
            ax = self.plt.subplot(nrow,ncol,1)

        md=task.meta
        expD=task.expD
        
        um=1e-6 # conversion from m to um
        
        # ......  Aquila FOV ......
        fov=md['payload']['aquila_conf']['area']
        fovX= float(fov['width'])/um
        fovY= float(fov['height'])/um
        nominal_Rb= md['payload']['nominal_Rb_um']

        
        ax.set_aspect(1.)
        mm=10
        doFOV=True
        if 'XrangeLR' in md:
            ax.set_xlim(tuple(md['XrangeLR']))
            doFOV=False
        else:
            ax.set_xlim(-mm,fovX+10);
        if 'YrangeLR' in md:
            ax.set_ylim(tuple(md['YrangeLR']))
            doFOV=False
        else:
            ax.set_ylim(-mm,fovX+10);
        

        if doFOV:
            ax.add_patch( self.plt.Rectangle([0,0],fovX,fovY, color="k", alpha=0.9, fc="none", ls='--') )

            ax.plot([fovX,fovX+3],[0,-3],color='k',ls='--')
            ax.text(fovX+2,-8,'FOV',color='k',va='bottom')
        
        ax.set(xlabel='X position ($\mu$m)',ylabel='Y position ($\mu$m)')
  
        if self.venue=='prod':
            tit='job=%s'%(md['short_name'])
            ax.set_title(tit)

            ax.grid(ls=':')
            #.... misc text
            txt1='back: '+md['submit']['backend']
            txt1+=', atoms: %d'%md['payload']['num_atom']
            txt1+=', omega: %.1f MHz'%md['payload']['rabi_omega_MHz']
            txt1+=', Rb: %.1f um'%nominal_Rb
            ax.text(0.02,0.95,txt1,color='g', transform=ax.transAxes)
        
        # ....  Atoms  ............
        atoms=task.atoms_pos_um()
        
        if atoms.size>0: # filled_sites
            ax.plot(atoms[:,0], atoms[:,1], 'r.', ms=11, label='filled')
        
        nA=atoms.shape[0]
    
        if what_to_draw=="circle":
            for i in range(nA):
                x,y=atoms[i]
                #  Rb radii
                ax.add_patch( self.plt.Circle((x,y), nominal_Rb/2, facecolor="b", alpha=0.3,edgecolor='none') )             
                                
                if self.venue=='prod':
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
    def global_drive(self,task,figId=4, axL=[]):
        if len(axL)==0:
            figId=self.smart_append(figId)
            nrow,ncol=2,1
            fig=self.plt.figure(figId,facecolor='white', figsize=(6,4))
            ax1,ax3 =  fig.subplots(2, 1, )
        else:
            ax1,ax3 = axL
            
        md=task.meta
        tV,ampls,detuns,phases=task.getDrive()
        #XtV,ampls,detuns,phases=task.expD['hamilt_drive']

        # adjust units
        us=1e-6 # units are seconds
        MHz2pi=2e6*np.pi  # units are (rad/sec)
        tV*=1e6
       

        ampls/=MHz2pi
        detuns/=MHz2pi
        phases/=MHz2pi
        
        # Plot y1 on the left axis
        line1 = ax1.plot(tV, detuns, 'b-', label=r'Detune $\Delta$')
        
        ax1.set_ylabel(r'$\Delta\,/\,2\pi$ (MHz)', color='b')
        
        # Create a twin y-axis on the right side
        ax2 = ax1.twinx()
        ax2.spines['right'].set_visible(True)

        # Plot y2 on the right axis
        line2 = ax2.plot(tV,ampls , 'r-', label=r'Rabi $\Omega$')
        ax2.set_ylabel(r'$\Omega\,/\,2\pi$ (MHz)', color='r')
        ax2.set_ylim(-0.3,2.8)
        
        # Set the y-axis label colors to match the data colors
        ax1.tick_params(axis='y', labelcolor=line1[0].get_color())
        ax2.tick_params(axis='y', labelcolor=line2[0].get_color())

        if self.venue=='paper':
            # Adding a yellow rectangle
            # The rectangle covers the area from x = 0 to 1, and y = 0 to 1
            rx1,rx2=tV[1],tV[3]
            ry1,ry2=detuns[1],detuns[3]
            rectangle = self.plt.Rectangle((rx1, ry1), (rx2-rx1), (ry2-ry1) , color='skyblue', alpha=0.8)
            ax1.add_patch(rectangle)

            return
        ax1.set_xlabel(r'Time ($\mu$s)')
        ax1.grid()
        ax1.set_title('Hamiltonian for job=%s'%(md['short_name']))
        
        # Add legend
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='center left')
        if 'TrangeLR' in md:
            ax1.set_xlim(tuple(md['TrangeLR']))
        
        if ax3!=None: # independent plot for phase
            ax3.plot(tV, phases, 'g-', label='phase')
            
            ax3.set_xlabel(r'Time ($\mu$s)')
            ax3.set_ylabel(r'$\Phi/2\pi$ (MHz)', color='g')
            ax3.legend( loc='upper right')    

    #...!...!....................
    def energy_spectrum(self,task,figId=3):
        figId=self.smart_append(figId)
        nrow,ncol=2,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(6,6))
        md=task.meta
        smd=md['submit']
        amd=md['postproc']
        pd=md['payload']
        nAtom= amd['num_atom']
        nominal_Rb= pd['nominal_Rb_um']

        #... get data
        
        energyV=task.expD['ranked_energy_eV']
        probV = task.expD['ranked_probs']
        hammwV=task.expD['ranked_hammw']

        K = 30 # bins for the histogram
        Ktic=4
        
        if 1:
            rx1,rx2=amd['energy_range_eV']
            binsX = np.linspace(0,K-1, Ktic)
            binsXtic=np.linspace(rx1,rx2, Ktic)
            xTicks = ['%.2g'%x for x in binsXtic]
            #print('sh:',probV.shape,energyV.shape)
            ry1,ry2=min(hammwV),max(hammwV)
            L=int(ry2-ry1)
            binsY= np.linspace(-0.,L-1,L)
            #print('yyy',binsY)

        
        #::::::::::::::::  1D density  ::::::::::::::::
        # Compute sum of y values per bin along x
        hist, bin_edges = np.histogram(energyV, bins=K, weights=probV[0])
        #print('hist:',hist)
        #print('edges:',bin_edges)
        ax = self.plt.subplot(nrow,ncol,1)
        ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges))

        tit='energy spectrum, %d atoms chain,  job=%s'%(nAtom,md['short_name'])
        ax.set(xlabel="final state energy (eV)",ylabel="probability sum",title=tit)
        ax.grid()
        #1ax.set_yscale('log')
        if 'eneRangeLR' in md:
            ax.set_xlim(tuple(md['eneRangeLR']))
 
        
        #::::::::::::::::  2D density (energy, hamming weight)  ::::::::::::::::
        # Compute sum of z values per bin along x and y
        
        #print('L',L)
        hist, x_edges, y_edges = np.histogram2d(energyV, hammwV, bins=[K, L], weights=probV[0])
        #print('hist:',hist)
        #print('xedges:',x_edges)
        #print('yedges:',y_edges)
        ax = self.plt.subplot(nrow,ncol,2)
        # Plot the 2D histogram
        image = ax.imshow(hist.T, origin='lower', cmap='YlOrBr')
        # Add a colorbar
        self.plt.colorbar(image, ax=ax, label='probability sum')
        ax.set(xlabel="final state energy (eV)",ylabel="hamm w.")

        # User-defined tick lists
        ax.set_xticks(binsX)
        ax.set_xticklabels(xTicks)

        ax.set_yticks(binsY)
        yTicks = ['%d'%(x) for x in y_edges[:-1]]
        ax.set_yticklabels(yTicks)
        ax.grid()

#...!...!....................
    def pattern_frequency(self,task,figId=3,ax=None):
        if ax==None:
            figId=self.smart_append(figId)
            nrow,ncol=1,1
            fig=self.plt.figure(figId,facecolor='white', figsize=(4,3))
            ax = self.plt.subplot(nrow,ncol,1)
        md=task.meta
        
        #... get data        
        probV = task.expD['ranked_probs'] # {prob,probEr,mshot} x {pattern index}
        mshotV=probV[2]
        print('sss',probV.shape)
        if 0: # hack to make histo look good if all patterns are different
            if int(max(mshotV))==1:  mshotV[0]=2

        print('dd',mshotV[:10])
        tit='job=%s'%(md['short_name'])        
        ax.set_ylim(0.7,)
        
        if 'maxNumOccur' in md:
            mxShot=md['maxNumOccur']
            ax.set_ylim(tuple(md['numStateRange']))
        else:
            mxShot=int(max(mshotV))
            
        # Creating a histogram of the random integers
        ax.hist(mshotV, bins=range(mxShot+1), align='left', rwidth=0.8, color='peru')

        # Creating the histogram of the random integers without edges
        #Xplt.hist(random_integers, bins=range(11), align='left', rwidth=0.8, color='blue')

        
        ax.set_yscale('log')
        ax.set(xlabel='Num. of occurences')
        ax.grid(axis='y', linestyle='--')
        if self.venue=='prod':            
            ax.set( ylabel='Num. of states')
            ax.set(title=tit)
        else:
            # Setting the y-axis labels to 1, 10, 100...
            ax.set_yticks([1, 10, 100, 1000], ['1', '10', '100','1000'])
            ax.set_xticks([1, 2,3,4,5,10,15,20], ['1', '2','3','4','5','10','15','20'])

 
