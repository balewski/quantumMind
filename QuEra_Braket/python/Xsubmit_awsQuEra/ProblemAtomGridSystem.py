__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import time
from pprint import pprint
import numpy as np
from decimal import Decimal
import json  # for saving program into bigD

from braket.ahs.atom_arrangement import AtomArrangement
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField
from braket.ahs.hamiltonian import Hamiltonian
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation

from toolbox.UAwsQuEra_job import harvest_submitInfo , retrieve_aws_job, harvest_retrievInfo, postprocess_job_results
from toolbox.Util_ahs import  compute_nominal_rb

from toolbox.Util_stats import do_yield_stats

#............................
#............................
#............................
class ProblemAtomGridSystem():  
    def __init__(self, args, jobMD=None, expD=None):
        self.verb=args.verb
        print('Task:',self.__class__.__name__)
        if jobMD==None:
            self._buildSubmitMeta(args)
        else:
            assert jobMD['payload']['class_name']==self.__class__.__name__
            self.meta=jobMD
            
        self.expD={} if expD==None else expD
           
    @property
    def submitMeta(self):
        return self.meta
    @property
    def numCirc(self):
        return len(self.circL)
        
#...!...!....................
    def _buildSubmitMeta(self,args):
        
        smd={'num_shots':args.numShots}
        smd['backend']=args.backendName
        
        # analyzis info
        amd={'ana_code':'ana_AGS.py'}

        pd={}  # payload
        pd['atom_dist_um']=Decimal(args.atom_dist_um)
        pd['evol_time_us']=Decimal(args.evol_time_us)
        pd['evol_time_ramp_us']=Decimal('0.1') 
                
        pd['rabi_omega_MHz']=args.rabi_omega_MHz
        pd['detune_MHz']=args.detune_MHz
        pd['num_atom']=args.num_atom
        pd['atom_in_column']=args.atom_in_column
        pd['grid_type']=args.gridType
        pd['class_name']=self.__class__.__name__
        #pd['info']='ini theta:%.1f phi:%.1f q:%s'%(pd['theta'],pd['phi'],csd['text_qubits'])
  
        md={ 'payload':pd,'submit':smd,'analyzis':amd}  # 'circ_sum':csd ,##
        md['short_name']=args.expName            

        self.meta=md
        if self.verb>1:
            print('BMD:');pprint(md)

#...!...!....................
    def placeAtoms(self):
        # hardcoded filed-of-view for Aquila
        AQUILA={'area': {'height': Decimal('0.000076'),  'width': Decimal('0.000075')},
                'geometry': {'numberSitesMax': 256,
                             'positionResolution': Decimal('1E-7'),
                             'spacingRadialMin': Decimal('0.000004'),
                             'spacingVerticalMin': Decimal('0.000004')}
                }
        
        M1=1000000
        xy00= 2* AQUILA['geometry'][ 'positionResolution']
        maxX=AQUILA['area']['width']-xy00
        maxY=AQUILA['area']['height']-xy00
        pd=self.meta['payload']
        register = AtomArrangement()
        gridDistY=pd['atom_dist_um']/M1 # in (m)
        if pd['grid_type']=='square':
            gridDistX=gridDistY
        if pd['grid_type']=='hex':
            gridDistX=gridDistY*Decimal('0.86') # *sqrt(3/4)
        nAtom=pd['num_atom']
        
        numY=pd['atom_in_column']
        print('place %d atoms\ni j  x(m)     y(m)    max X,Y:'%nAtom,maxX,maxY) 
        for k in range(nAtom):
            i=k//numY ; j=k%numY
            x=xy00+i*gridDistX ;  y=xy00+j*gridDistY
            if  pd['grid_type']=='hex' and i%2==0 :  # reverse y-dir for even columns
                #y=maxY-y
                y=y+gridDistY/2

            assert x>=0 ; assert x<=maxX
            assert y>=0 ; assert y<=maxY
            print(i,j,x,y,x<=maxX, y<maxY)
            register.add([x,y])

        self.register=register
 
        #... update meta-data
        self.meta['payload']['aquila_area']=AQUILA['area']
        # atoms placement to dataD
        xL=register.coordinate_list(0)
        yL=register.coordinate_list(1)

        xD=[[x,y] for x,y in zip(xL,yL)]
        self.expD['atom_xy.JSON']=json.dumps(xD, default=str) # Decimal --> str
        
        if self.verb<=1: return

        print('\ndump atoms coordinates')
        for i in range(nAtom):
            x=xL[i]*M1
            y=yL[i]*M1
            print('%d atom : x=%.2f y= %.1f (um)'%(i,x,y))  
    
#...!...!....................
    def buildHamiltonian(self):
        M1=1000000
        pd=self.meta['payload']
        t_max  =pd['evol_time_us']/M1
        t_ramp =pd['evol_time_ramp_us']/M1

        assert t_max>= 2* t_ramp+ t_ramp  # for simpler interpretation
        
        omega_min = 0       
        omega_max = pd['rabi_omega_MHz'] * 2 * np.pi *1e6  # units rad/sec
        delta_max=0           # no detuning

        delta_ampl=pd['detune_MHz'] * 2 * np.pi *1e6  # units rad/sec
        delta_ampl1=int(delta_ampl/400)*Decimal('400') 
        # constant Rabi frequency
       
        t_points = [0, t_ramp, t_max - t_ramp, t_max]
        omega_values = [omega_min, omega_max, omega_max, omega_min]
        Omega = TimeSeries()
        Delta_glob = TimeSeries()
        for t,v in zip(t_points,omega_values):
            bb=int(v/400)*Decimal('400')  
            Omega.put(t,bb)
            Delta_glob.put(t,delta_ampl1)
            
        # all-zero phase and detuning Delta
        Phase = TimeSeries().put(0.0, 0.0).put(t_max, 0.0)  # (time [s], value [rad])
        #Delta_global = TimeSeries().put(0.0, 0.0).put(t_max, delta_max)  # (time [s], value [rad/s])

        drive = DrivingField(
            amplitude=Omega,
            phase=Phase,
            detuning=Delta_glob
        )
        H = Hamiltonian()
        H += drive
        self.H=H
        pd['nominal_Rb_um']=compute_nominal_rb(omega_max/1e6, delta_ampl/1e6 )
        if self.verb<=1: return

        print('\ndump drive filed Amplitude(time) :\n',drive.amplitude.time_series.times(), drive.amplitude.time_series.values())

        
#...!...!....................
    def buildProgram(self):
        pd=self.meta['payload']
        ahs_program = AnalogHamiltonianSimulation(
            hamiltonian=self.H,
            register=self.register
        )

        # program to dataD
        circD=ahs_program.to_ir().dict()
        self.expD['program_org.JSON']=json.dumps(circD, default=str) # Decimal --> str
       
        if  self.verb>1:
            print('\ndump Schrodinger problem:')           
            pprint(circD)
            
        return ahs_program

#...!...!..................
    def postprocess_submit(self,job):        
        harvest_submitInfo(job,self.meta,taskName='tas') # -->ags
        

#...!...!..................
    def retrieve_job(self,job=None):
        
        isBatch= 'batch_handles' in self.expD  # my flag
        
        if job==None:
            smd=self.meta['submit']  # submit-MD
            arn=smd['task_arn']
            #arn='arn:aws:braket:us-east-1:765483381942:quantum-task/874bf405-8720-4e77-9fec-b6b84bfa5016' #   'status': 'COMPLETED'
            #arn='arn:aws:braket:us-east-1:765483381942:quantum-task/1830c574-3744-44f8-8bee-9af4e70a0757'  # state CANCELLED 
            job = retrieve_aws_job( arn, verb=self.verb)
                
            print('retrieved ARN=',arn)
        if self.verb>1: print('job meta:'); pprint( job.metadata())
        result=job.result()
        
        # res: <class 'braket.tasks.analog_hamiltonian_simulation_quantum_task_result.AnalogHamiltonianSimulationQuantumTaskResult'>
        # https://amazon-braket-sdk-python.readthedocs.io/en/stable/_apidoc/braket.tasks.analog_hamiltonian_simulation_quantum_task_result.html
        
        print('res:', type(result))
       
        t1=time.time()    
        harvest_retrievInfo(job.metadata(),self.meta)
        
        if isBatch:
            never_tested22
            jidL=[x.decode("utf-8") for x in expD['batch_handles']]
            print('jjj',jidL)
            jobL = [backend.retrieve_job(jid) for jid in jidL ]
            resultL =[ job.result() for job in jobL]
            jobMD['submit']['cost']*=len(jidL)
        else:
            rawBitstr=job.result().get_counts()
            #print('tt',type(rawBitstr))
            #pprint(rawBitstr)
            
        t2=time.time()
        print('retriev job(s)  took %.1f sec'%(t2-t1))

        postprocess_job_results(rawBitstr,self.meta,self.expD)

#...!...!..................
    def _analyze_heralding_per_atom(self,rawBitstr):
        nAtom=self.meta['payload']['num_atom']
        NB=2 # it is always per-atom analysis
        dataY=np.zeros((nAtom,NB),dtype=np.int32)
        #print('tt2',type(rawBitstr))
        for key in rawBitstr:
            val=rawBitstr[key]
            #print('\n key',key)
            for j in range(nAtom):
                s=key[j]
                i=0  if s=='e' else 1  # map ee->0, g|r->1
                #print(j,s,i,'val=',val,  s=='e' )
                dataY[j,i]+=val
            #break
        #1print('Herald for %d atoms:'%nAtom); pprint(dataY)
        
        dataP=do_yield_stats(dataY)

        self.expD['herald_counts_atom']=dataY # shape[nAtom,NB]
        self.expD['herald_prob_atom']=dataP  # shape[PAY,nAtom]

        
#...!...!..................
    def analyzeExperiment(self):
        import json # tmp for raw data  unpacking
        rawBitstr=json.loads(self.expD.pop('counts_raw.JSON')[0])
        if len(rawBitstr) <15:
            print('dump rawBitstr:'); pprint(rawBitstr)

        self._analyze_heralding_per_atom(rawBitstr)

        # process every atom independently
        nAtom=self.meta['payload']['num_atom']
        NB=2 # rabi-measurement  is  per-atom analysis, atoms are independent
        #print('NB:',NB,nAtom); a1
        rgStates=['g','r']
        # for more generic mult-atom bit-strings see problem_BellState.analyzeExperiment()
        cnt={ j:{'e':0,'g':0,'r':0}  for j in range(nAtom)}
        #.... update meta-data
        amd=self.meta['analyzis']
        amd['num_rg_states']=NB
        amd['rg_states_list']=rgStates  # order as stored in self.expD['prob_clust']
        
        t_max=self.meta['payload']['evol_time_us']
        a_dist=self.meta['payload']['atom_dist_um']
        omega_MHz= self.meta['payload']['rabi_omega_MHz']
        for key in rawBitstr:
            val=rawBitstr[key]
            #print('key:',key,val)
            for j in range(nAtom):
                s=key[j]
                cnt[j][s]+=val
        print('\nEvolution time=%s us, dist=%s um, Omega=%.2f MHz, job: %s'%(t_max,a_dist,omega_MHz,self.meta['short_name']))
        print('%d atoms:'%nAtom); pprint(cnt)
        
                    
        def counts2numpy(num_atom,egrCnt):
            counts=np.zeros((num_atom,NB),dtype=np.int32)
            #print('pp',counts.shape)
            for j in egrCnt:
                rec=egrCnt[j]
                counts[j][0]=rec['g']
                counts[j][1]=rec['r']
            #print('pp2',counts)
            return counts
        
        dataY=counts2numpy(nAtom,cnt)
        dataP=do_yield_stats(dataY)

        self.expD['counts_atom']=dataY # shape[nAtom,NB]
        self.expD['prob_atom']=dataP

        
        # sum all atoms
        dataYS=np.sum(dataY,axis=0).reshape(1,-1)
        dataPS=do_yield_stats(dataYS)
        self.expD['counts_sum']=dataYS.astype('int32')
        self.expD['prob_sum']=dataPS
        
        if nAtom!=16:
            print("SKIP per-column analysis for nAtom=%d"%nAtom)
            return
                  
        # sum atoms by column
        colSize=self.meta['payload']['atom_in_column']
        dataYcol=np.sum(dataY.reshape(colSize,-1,NB),axis=0) # shape[nCol,NB]
        dataPcol=do_yield_stats(dataYcol)
        self.expD['counts_col']=dataYcol.astype('int32')
        self.expD['prob_col']=dataPcol
        
        # sum atoms by row
        dataYrow=np.sum(dataY.reshape(colSize,-1,NB),axis=1) # shape[nRow,NB]
        dataProw=do_yield_stats(dataYrow)
        self.expD['counts_row']=dataYrow.astype('int32')
        self.expD['prob_row']=dataProw
                
       
