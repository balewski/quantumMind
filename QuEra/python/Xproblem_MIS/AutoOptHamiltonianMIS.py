import time
import numpy as np
from pprint import pprint
import os

from ProblemMaxIndependentSet import ProblemMIS
from toolbox.UAwsQuEra_job import  access_quera_device
from toolbox.UAwsQuEra_job import  harvest_retrievInfo, postprocess_job_results 
from decimal import Decimal
from collections import Counter
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

from ana_MIS import Plotter
class EmptyClass:  pass

class AutoOptHamiltonianMIS(ProblemMIS):
#...!...!..................
    def __init__(self,args):  #
        
        ProblemMIS.__init__(self,args)
        self.placeAtoms()        
        self.autoPath=args.outPath
        self.num_detune_par=args.num_detune_par
            
        if  args.backendName=='emulator' :
            self.device=access_quera_device(args.backendName,verb=args.verb)
  
        if 0:           
            tc=self.taskConf
            expN=tc['task_name']

            #pprint(tc)
            self.xbinV_name=expN+'_twidth'
            self.summaryD['configTask']=self.taskConf  # updated be each experiment
            self.summaryD['configCommon']=self.comConf
        self.cnt={'run':0}
        self.Tstart=time.time()

        
#...!...!..................
    def run_one_experiment(self,inpD):
        nDetune=self.num_detune_par
        print("\n\n- - - - - - - - - OE:begin new experiment",self.cnt,"elaT=%.1f min"%((time.time()-self.Tstart)/60.))
        print('ROE: inpD:'); pprint(inpD)
        pd=self.meta['payload']

        #transfer inpD to Hamiltonian

        t_up=inpD['ramp_up_us']
        t_down=inpD['ramp_down_us']
        pd['rabi_ramp_time_us']=[t_up,t_down]
        pd['detune_shape']=[ inpD['detune_a%d'%i] for i in range(nDetune) ]
        
        self.buildHamiltonian()
        ahs_program=self.buildProgram()  # SchrodingerProblem
        
        smd=self.meta['submit']
        evolSteps=int(pd['evol_time_us']*100)
        shots=smd['num_shots']
        print('\n Emulator: evol time %.3f us , steps=%d'%(float(pd['evol_time_us']),evolSteps))
        
        # Set a different seed based on the current time
        np.random.seed(int(time.time()))

        job = self.device.run(ahs_program, shots=shots, steps=evolSteps)
        self.postprocess_submit(job.metadata())

        harvest_retrievInfo(None,self.meta)
        rawCounts=job.result().get_counts()
        postprocess_job_results(rawCounts,self.meta,self.expD)
        
        if len(rawCounts)<10:
            print("A:rawCounts:")
            pprint(rawCounts)
        self.rawCounter=Counter(rawCounts)
        self.analyzeExperiment()

        #print('A:ranked_probs',self.expD['ranked_probs'].shape)
        #print('A:ranked_MISs\n',self.expD['ranked_MISs'])
        #pprint(pd); ok11

      
        self.save_iteration()
        return
        # self.create_data_storageMultiQ([qqid],NX,doHerald=True)
        
        # ...  tmp save rawIQ
        expN=self.getTaskName()
        self.bigData[expN+'_rawIQ']=self.data_rawMIQ
        #  ['NY', 'NX', 'NS', 'NQ', 'NM', 'IQ']
        rawIQ= np.squeeze(self.data_rawMIQ) # skip NY,NQ-dims: all axis with dim=1


        #. . . . . . fit GMM  in-fly
        ampl_fact=1.e3  # convenience factor for the disply, but keep it here for consistency
        
        rawIQn=rawIQ/ampl_fact  # needed also during digitizartion - must match GMM
        HM,IQ2=rawIQn.shape[-2:]
        flatIQ=rawIQn.reshape(-1,HM,IQ2)  # clones data
        print('OE:flat HMxIQ',flatIQ.shape)
        heraIQ=flatIQ[:,0]
        measIQ=flatIQ[:,1]

        #... do 3-cluster GMM fit only to get BIC
        gmix3=GaussMix2D(num_clust=3)
        gmix3.fit(measIQ)
        bic_3=gmix3.myBIC
        
         #... do 1-cluster GMM fit only to get BIC
        gmix1=GaussMix2D(num_clust=1)
        gmix1.fit(measIQ)
        bic_1=gmix1.myBIC

        #... do 4-cluster GMM fit only to get BIC
        gmix4=GaussMix2D(num_clust=4)
        gmix4.fit(measIQ)
        bic_4=gmix4.myBIC
        
        #... do 2-cluster GMM fit for everything
        gmix=GaussMix2D(num_clust=2)
        gmix.fit(measIQ)
        
        auxD=gmix.lock_qubit_state0(heraIQ)
        auxD['BIC_2']=gmix.myBIC
        auxD['BIC_3']=bic_3
        auxD['BIC_4']=bic_4
        auxD['BIC_1']=bic_1
        print("OE:gmm auxD",auxD)
        self.summaryD['gmmSummary']=auxD
        
        # ....  fit rabi osc
        gmix.anaMD={} #hack, need just a pointer instead of args.
        oscModel=ModelSin()
        xBins=xbinV*1.e9 # needed by fitter & plotter
        fitData=fit_data_model(gmix,probData,xBins, oscModel )
        self.summaryD['fitSummary']=gmix.anaMD      

        self.summaryD['anaSummary']={}
        

#...!...!..................
    def save_iteration(self):
        print('saving data ...')
        self.makeSummary()        
        short_name=self.getShortName()+'_it%d'%self.cnt['run']
        
        self.meta['optimizer_iteration']=self.cnt['run']

        outF=short_name+'.h5'
        write4_data_hdf5(self.expD,os.path.join(self.autoPath,outF),self.meta, verb=0)
        
        fargs=EmptyClass
        fargs.noXterm=True
        fargs.verb=1
        fargs.outPath=self.autoPath
        plot=Plotter(fargs,short_name)
        plot.MIS_solutions( self,nSol=4 )
        plot.global_drive(self)
        plot.display_all()
              
        self.cnt['run']+=1        
    
       
        #
#...!...!..................
    def makeSummary(self):
        #print('MSR:',sorted(self.comConf))
        tmpD={}
        #for x in ['auxil',  'chip_gates_name', 'chip_hfdriver_map_name', 'chip_name', 'fpga_clock_step', 'fpga_ip',  'fpga_registers_map_name', 'fpga_wave_group_name', 'fridge_name', 'is_simu', 'measure_conf_name']:
        #    tmpD[x]=self.comConf[x]

        #self.summaryD['configCommon']=tmpD
        #self.summaryD['configTask']=self.taskConf  # updated be each experiment
        #self.summaryD['executionSummary']=self.execSum # appened info while executing 
        a=1
        
