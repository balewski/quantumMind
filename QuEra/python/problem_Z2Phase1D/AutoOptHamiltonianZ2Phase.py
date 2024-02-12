import time
import numpy as np
from pprint import pprint
import os

from ProblemZ2Phase import ProblemZ2Phase
from toolbox.UAwsQuEra_job import  access_quera_device
from toolbox.UAwsQuEra_job import  harvest_retrievInfo, postprocess_job_results 
from decimal import Decimal
from collections import Counter
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

from ana_Z2Phase import Plotter
class EmptyClass:  pass

class AutoOptHamiltonianZ2Phase(ProblemZ2Phase):
#...!...!..................
    def __init__(self,args):  #        
        ProblemZ2Phase.__init__(self,args)
        self.meta['payload']['class_name']='ProblemZ2Phase' # for 2pt-corr to work
        #pprint(self.meta); ok99
        
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

        #....transfer inpD to register & Hamiltonian
        t_up=inpD['t_ramp_up_us']
        t_down=inpD['t_ramp_down_us']
        pd['rabi_ramp_time_us']=[t_up,t_down]
        pd['detune_shape']=[ inpD['detune_a%d'%i] for i in range(nDetune) ]
        pd['atom_dist_um']=Decimal('%.7f'%inpD['atom_dist_um'])  #0.1um resolution 
        
        self.placeAtoms()    
        self.buildHamiltonian()
        ahs_program=self.buildProgram()  # SchrodingerProblem
        
        smd=self.meta['submit']
        evolSteps=int(pd['evol_time_us']*100)
        shots=smd['num_shots']
        print('\n Emulator: evol time %.3f us , steps=%d'%(float(pd['evol_time_us']),evolSteps))
        
        job = self.device.run(ahs_program, shots=shots, steps=evolSteps, solver_method="bdf")
        self.postprocess_submit(job.metadata())

        harvest_retrievInfo(None,self.meta)
        rawCounts=job.result().get_counts()
        postprocess_job_results(rawCounts,self.meta,self.expD)
        
        if len(rawCounts)<10:
            print("A:rawCounts:")
            pprint(rawCounts)
        self.rawCounter=Counter(rawCounts)
        self.analyzeRawExperiment()
        self.energySpectrum()
        self.twoPointCorrelation()

        #print('A:ranked_probs',self.expD['ranked_probs'].shape)
        #print('A:ranked_MISs\n',self.expD['ranked_MISs'])
        #pprint(pd); ok11

      
        self.save_iteration()
        return
    
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
        fargs.verb=0
        fargs.outPath=self.autoPath
        plot=Plotter(fargs,short_name)
        plot.global_drive(self,figId=2)
        plot.Z2Phase_solutions( self,nSol=6 ,figId=3)
        plot.correlations(self,'density',figId=10)
        plot.energy_spectrum(self,figId=12)
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
        
