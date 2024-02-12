__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

#  Fitter class anaylyzing Zero Noise Extrapolation (ZNE) data 

import os,time # for rnd seed
import numpy as np
from pprint import pprint
from toolbox.PlotterBackbone import PlotterBackbone
from toolbox.Util_Fitter import lmfit2dict , fit_data_model

#............................
#............................
#............................
class FitterReadErrZNE(PlotterBackbone):  # does also plotting
    def __init__(self,args):
        args.prjName=args.expName
        PlotterBackbone.__init__(self,args)
        
#...!...!..................
    def fit_model(self,dataP,xBins, oscModel):
    
        fit_osc,fitMD,fitResult=fit_data_model(dataP,xBins, oscModel)
        fitMD['xunit']=r'noise factor $\lambda$'
        fitMD['yunit']='avr. wall density number'
        
        #.... extrapolate model ....
        xL=0; xR=np.max(xBins)
        M=10
        xBins2 = np.linspace(xL, xR, M+1)
        fit_extra = oscModel.eval(params=fitResult.params, x=xBins2)
         
        bigD={'dataProb':dataP,'xBins':xBins,'dataFit':fit_osc,'extraY':fit_extra,'extraX':xBins2}
        self.data=bigD
        self.fitMD=fitMD
        self.fitModel=oscModel
        #1print('fit MD:'); pprint(fitMD)
        bigD['extraYerr']=self._extrapolate_fit_using_covariance(fitResult)



#...!...!..................
    def _extrapolate_fit_using_covariance(self,fitResult):
        isLinear='linear' in str(fitResult.model)
        isQuad='quad' in str(fitResult.model)
        print('lin,parab:',isLinear,isQuad)
        sigF=[]
        for new_x in  self.data['extraX']: 
            print('EFCV: new_x:',new_x)
            # Calculate the predicted Y for the new X value
            predicted_y = fitResult.eval(x=new_x)
            print('new Y:',predicted_y)
            # Get the covariance matrix of the fit parameters
            cov_matrix = fitResult.covar
            # Calculate the gradient vector
            if isLinear:  gradient_vector = np.array([ new_x, 1])
            if isQuad:  gradient_vector = np.array([ new_x**2, new_x, 1])
        
            # Calculate the standard error of the predicted Y for the new X
            std_error_predicted_y = np.sqrt(np.dot(np.dot(gradient_vector, cov_matrix), gradient_vector))
        
            print(f"Predicted Y for X = {new_x}: {predicted_y}")
            print(f"Standard Error of Predicted Y: {std_error_predicted_y}")
            sigF.append(std_error_predicted_y)
            
        return np.array(sigF)
        
#...!...!..................
    def draw_ZNEfit(self,expMD,expD,figId=7):
        figId=self.smart_append(figId)
        
        bigD=self.data
        fmd=self.fitMD
        pmd=expMD['payload']
        zmd=expMD['zne_wall_dens']
        isIdealSample='ideal_sample' in pmd
        
        inX=5;inY=6; ncol=1
        isPaper='paper' in self.venue
        fig=self.plt.figure(figId,facecolor='white',figsize=(inX,inY))
        nrow,ncol=1,ncol

        xBins=bigD['xBins']
        dataP=bigD['dataProb']
        xunit=fmd['xunit']
        yunit=fmd['yunit']       
 
        # .... LEFT PLOT ... raw data vs. noise scale 

        ax = self.plt.subplot(nrow, ncol, 1)
        extrX=bigD['extraX']
        extrY=bigD['extraY']
        extrE=bigD['extraYerr']
        ax.plot(extrX, extrY,label=fmd['fitFuncName'], color='blue',lw=1.)  
        ax.plot(extrX, extrY+extrE,'g--',label='std dev ZNE',lw=0.7)
        ax.plot(extrX, extrY-extrE,'g--',lw=0.7)
        ax.plot(extrX[0], extrY[0],ms=10,label='ZNE value', color='blue',  markerfacecolor='none', marker='o')

        
        k0=0 ; k1=k0+1           
        # ....measured data
        ax.errorbar(xBins[k0],dataP[0][k0],yerr=dataP[1][k0],fmt='o',markersize=4,label='measurment',color='red')
        
        # .... noise ammpliffied data        
        ax.errorbar(xBins[k1:],dataP[0][k1:],yerr=dataP[1][k1:],fmt='D',markersize=4,label='noise added',color='blue')

        if isIdealSample:
            momentV=expD["domain_wall_dens_moments"]
            xTrue=0.
            yTrue,yTrueE=momentV[0][:2]
            # ... ground truth
            #1ax.errorbar(xTrue, yTrue,yerr=yTrueE,fmt='*',markersize=8,label='ground truth',color='green')
            ax.plot(xTrue, yTrue,'*',markersize=9,label='ground truth',color='m',alpha=0.7)
            ax.set_ylim(min(yTrue,extrY[0])-0.5,)
     
        ax.set(xlabel=xunit,ylabel= yunit)
        
        ax.grid()
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=3, mode="expand", borderaxespad=0., )
    
        # Set the maximum number of ticks without specifying their positions
        self.plt.locator_params(axis='x', nbins=4)

       
        # - - - - -  decorations for fit-plot
        txt=r'$\leftarrow$ ZNE value: %.2f +/-%.2f '%(extrY[0],extrE[0])
        ax.text(extrX[2],extrY[0],txt, color='blue', va='center')
        #print('gold:',txt)
        #print('gold,%.2f,%.2f'%(extrY[0],extrE[0]))
        
     
        smd=expMD['submit']
        pmd=expMD['payload']
        fqa=self.fitMD['fit_result']['fitQA']
        txt='job: %s '%(expMD['short_name'])
        txt+='\nbackend: %s, %s'%(smd['backend'],smd['date'][:-4])
        txt+='\nshots/task: %d, resample: %d  '%(smd['num_shots'],zmd['sample_reuse'])
        txt+='\n%s  nfree: %d   redchi: %.2f'%(self.fitMD['fitFuncName'],fqa['nfree'][0],fqa['redchi'][0])
        txt+='\nreadErr s0m1: %.2f  s1m0: %.2f'%(zmd['readout_error'][0],zmd['readout_error'][1])       

        x0=0.02; y0=0.8
        ax.text(x0,y0,txt,color='m',transform=ax.transAxes)

        #print('left info:',txt)
   

        
    #...!...!....................
    def draw_domain_wall_density(self,md,bigD,figId=1):
        
        pmd=md['payload']
        zmd=md['zne_wall_dens']
        
        isIdealSample='ideal_sample' in pmd
        nAtom= md['postproc']['num_atom'] 

        dens2D=bigD["domain_wall_dens_histo"]
        noiseV=bigD["SPAM_scale"]
        momentV=bigD["domain_wall_dens_moments"]
        K=noiseV.shape[0]
        densX = np.arange(dens2D.shape[1])

        if isIdealSample:
            k0=0
            tit0='ideal sample'            
            
        else:
            k0=1
            tit0=md['short_name']
            K+=1
            
            
        tLab='ReadErr ZNE %s (%d-atoms)'%(tit0,nAtom)       
       
        nrow,ncol=K,1        
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(5,10))
        axs= fig.subplots(nrow,ncol,sharex=True)

        axs[0].set_title(tLab)
        
        for k in range(k0,K):
            ax=axs[k]
            j=k-k0
            
            if k==0:
                dCol='gold'; tit1=tit0+' , ground truth'                
            elif k==1:
                dCol='lime' ; tit1='measured data'
            else:
                dCol='blue' ; tit1='ampliffied noise'
            # Create a bar plot to visualize the histogram
            ax.bar(densX,dens2D[j] , width=1., edgecolor='black', color=dCol)

            ax.set_ylabel('probability')
            ax.grid()
                
            if k==K-1: # bottom row
                ax.set(xlabel="wall density number")           
                
            #  beautification
            avrX,avrXer,stdX,stdstd,mshots=momentV[j]
            yy=0.0
            ax.errorbar([avrX], [yy] , xerr=avrXer,fmt='D',color='r',markersize=10,linewidth=0.8)
            txt1=tit1
            
            txt1+='\nSPAM scale: %.1f'%(noiseV[j])
            txt1+='\navr %.2f +/-%.2f  \nstd=%.1f  shots=%d'%(avrX,avrXer,stdX,mshots)
            txt1+='\nused readErr:\n   s0m1=%.2f  s1m0=%.2f'%(zmd['readout_error'][0],zmd['readout_error'][1])            
            ax.text(0.5,0.2,txt1,transform=ax.transAxes,color='b')
            
            if 'densLR' in md:   ax.set_xlim(tuple(md['densLR']))
           
