__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import time
from pprint import pprint
import numpy as np
from decimal import Decimal
import json  # for saving AHS program into bigD
import networkx as nx  # for graph generation
from collections import Counter
from bitstring import BitArray

from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.hamiltonian import Hamiltonian
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation 

from toolbox.UAwsQuEra_job import harvest_submitInfo , retrieve_aws_job, harvest_retrievInfo, postprocess_job_results, hacked_get_shots, sort_subset_of_shots
from toolbox.Util_ahs import compute_nominal_rb, register_to_numpy, drive_to_numpy, register_from_numpy, drive_from_numpy, ramp_drive_waveform, mid_drive_waveform, scar_drive_waveform, raw_experiment_postproc, states_energy
from toolbox.Util_miscIO import  graph_from_JSON, graph_to_JSON
from toolbox.Util_stats import do_yield_stats

from Util_Z2Phase import connected_two_point_corr_func_v2, dens_integrand, do_magnetization, fit_exponent_simple

#............................
#............................
#............................
class ProblemZ2Phase():  # Study Z2 phase in 1D ordered Rydberg system
#...!...!....................
    def __init__(self, args, jobMD=None, expD=None):
        self.verb=args.verb
        print('Task:',self.__class__.__name__)
        if jobMD==None:
            self._buildSubmitMeta(args)
            self.expD={}
        else:
            assert jobMD['payload']['class_name']==self.__class__.__name__
            if args.useHalfShots !=None:
                khalf=args.useHalfShots                
                print('PZP: use %d-half of  shots'%khalf)
                sort_subset_of_shots(khalf,jobMD,expD)

            if 'ranked_hexpatt' in expD:  #... recover bitstrings
                hexpattV=expD['ranked_hexpatt']
                #print('rr',hexpattV.shape)
                k=hexpattV.shape[0]
                for i in range(k):                    
                    expD['ranked_hexpatt'][i]=hexpattV[i].decode("utf-8")
                    
            self.meta=jobMD
            self.expD=expD
            self._ahs_problem_from_bigData()    
           
    @property
    def submitMeta(self):       return self.meta
    @property
    def numCirc(self):          return self.submitMeta['short_name']
    def getShortName(self):     return self.submitMeta['short_name']
    def setShortName(self,txt): self.submitMeta['short_name']=txt
    def atoms_pos_um(self):     return self.expD['atoms_xy']*1e6
    def getDrive(self):
        times,amplitudes,detunings,phases=self.expD['hamilt_drive']
        return times,amplitudes,detunings,phases
    
#...!...!....................
    def _ahs_problem_from_bigData(self):
        bigD=self.expD
        self.register=register_from_numpy(bigD['atoms_xy'])
        self.H=drive_from_numpy(bigD['hamilt_drive'])
        self.graph=graph_from_JSON(bigD['graph.JSON'][0])
        # -------  process counts - if any  -----
        if 'counts_raw.JSON' in bigD:  
            rawBitstr=json.loads(bigD['counts_raw.JSON'][0])
            #print('RBS:',rawBitstr)
            self.rawCounter=Counter(rawBitstr)
            if len(rawBitstr) <17 :
                print('dump rawBitstr:'); pprint(rawBitstr)
#...!...!....................
    def _ahs_problem_to_bigData(self):
        bigD=self.expD
        bigD['atoms_xy']=register_to_numpy(self.register)
        bigD['hamilt_drive']=drive_to_numpy(self.H.terms[0]); #the same as drive
        bigD['graph.JSON']=graph_to_JSON(self.graph)

    
#...!...!....................
    def _buildSubmitMeta(self,args):      
        smd={'num_shots':args.numShots}
        smd['backend']=args.backendName
        
        # postprocessing
        ppmd={'post_code':'postproc_Z2Phase.py'}

        pd={}  # payload
        pd['atom_dist_um']=Decimal(args.atom_dist_um)
        pd['chain_shape']=args.chain_shape
        pd['evol_time_us']=args.evol_time_us
        pd['rabi_ramp_time_us']=args.rabi_ramp_time_us
        pd['detune_shape']=args.detune_shape
        pd['rabi_omega_MHz']=args.rabi_omega_MHz
        pd['detune_delta_MHz']=args.detune_delta_MHz
        pd['scar_time_us']=args.scar_time_us
        pd['hold_time_us']=args.hold_time_us
        pd['class_name']=self.__class__.__name__
        
        pd['num_clust']=-1 if   args.multi_clust else 1 #tmp , see placeAtoms(.)
        md={ 'payload':pd,'submit':smd,'postproc':ppmd} 
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
        pd=self.meta['payload']
        pd['aquila_conf']=AQUILA

        M1=1000000
        separation=pd['atom_dist_um']/M1 # in (m)
        
        ctype, natom=pd['chain_shape']
        nAtom=int(natom)
        register=None
        if ctype=='line':  
            register, G = generate_line(nAtom, self.meta, dist=separation)
        if ctype=='uturn':  
            register, G = generate_uturn_line(nAtom, self.meta, dist=separation, xoffset=Decimal('0'))
        if 'loop' in ctype:  
            register, G = generate_closed_loop(nAtom, self.meta, ctype)

        assert register!=None, 'wrong chain name?'

        # define target state before replication  of the cluster or ancillas
        hexlen=int(np.ceil(nAtom / 4) * 4)
        pre0=hexlen-nAtom
        A=BitArray(length=pre0)
        B=BitArray(length=nAtom)
        for i in range(0,nAtom,2): B[i]=1
        
        if 0: # change objective is no 000 in the middle
            iMid=nAtom//2
            assert iMid%2==0 ,'otherwise 0 is not in the middle' 
            B[iMid]=0  # makes 000 pattern
        
        C=A+B
        #print('AA',nAtom,C.bin)
        print('target state hex',C.hex)
        pd['target_state_hex']=C.hex

        
        if pd['num_clust']==1:
            pd['num_atom_in_clust']=len(register)
        else:            
            register=self._replicateClucters(register)
        
        self.register=register
        self.graph=G
        nAtom=len(register)

        #... update meta-data
        pd['num_atom']=nAtom
        pd['info']='atoms:%d,omega:%.1f,dist:%.1f'%(nAtom,pd['rabi_omega_MHz'],pd['atom_dist_um'])
        if self.verb<=1: return

        print('\ndump atoms coordinates')
        pos=register_to_numpy(register)
        for i in range(nAtom):
            print('%d atom : x=%.2f y= %.1f (um)'%(i,pos[i,0],pos[i,1]))  
    
#...!...!....................
    def buildHamiltonian(self):
        um=1e-6 # units are meters
        us=1e-6 # units are seconds
        MHz2pi=2e6*np.pi  # units are (rad/sec)
        M1=1000000
        pd=self.meta['payload']
        omega_max = pd['rabi_omega_MHz'] * MHz2pi # units rad/sec

        delta_begin = pd['detune_delta_MHz'][0] * MHz2pi # units rad/sec
        delta_end   = pd['detune_delta_MHz'][1] * MHz2pi # units rad/sec

        t_max  =pd['evol_time_us']/M1  # in sec
        t_up =pd['rabi_ramp_time_us'][0]/M1
        t_down =pd['rabi_ramp_time_us'][1]/M1
        t_vary=t_max - t_up - t_down
        #print('ttt',  t_vary,t_max,  t_up,  t_down)
        assert t_vary>=0.1e-6 
        Ha=ramp_drive_waveform(t_up,omega_max,delta_begin,'pre')        
        Hb=mid_drive_waveform(t_vary,omega_max,delta_begin,delta_end,pd['detune_shape'] )
        Hc=ramp_drive_waveform(t_down,omega_max,delta_end,'post')
        H=Ha.stitch(Hb).stitch(Hc)

        #.... add WMBS evolution, if any
        tscarL=pd['scar_time_us']        
        if len(tscarL)>0:
            omega_half=omega_max/2
            t_ramp=tscarL[0]/M1  # in sec
            t_flat=tscarL[1]/M1  # in sec
            Hd=scar_drive_waveform(t_ramp,t_flat,omega_max/2,delta_end)
            H=H.stitch(Hd)

        #... delay read , if any
        if pd['hold_time_us']>0.05:  # (us)
            t_delay=pd['hold_time_us']/M1  # in sec
            He=mid_drive_waveform(t_delay,0,delta_end,delta_end,[0.5] )
            H=H.stitch(He)
            
        if self.verb>1:  print('BH:',drive_to_numpy(H))
        self.H=H
        pd['nominal_Rb_um']=compute_nominal_rb(omega_max, 0.)*1e6


#...!...!....................
    def buildProgram(self):
        pd=self.meta['payload']
        ahs_program = AnalogHamiltonianSimulation(
            hamiltonian=self.H,
            register=self.register
        )
        self.program=ahs_program                
        #self._body_to_bigData()
        self._ahs_problem_to_bigData()
            
        return ahs_program

#...!...!..................
    def postprocess_submit(self,job):        
        harvest_submitInfo(job,self.meta,taskName='z2ph')
                

#...!...!..................
    def retrieve_job(self,job=None):        
        isBatch= 'batch_handles' in self.expD  # my flag, not used yet
        
        if job==None:
            smd=self.meta['submit']  # submit-MD
            arn=smd['task_arn']
            job = retrieve_aws_job( arn, verb=self.verb)
                
            print('retrieved ARN=',arn)
        if self.verb>1: print('job meta:'); pprint( job.metadata())
        result=job.result()
        
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
            #print('tt',type(rawBitstr));  #pprint(rawBitstr)
            rawShots=hacked_get_shots(job.result().measurements)
            print('rawShots len=%d, some:'%len(rawShots),rawShots[:3])
                        
        t2=time.time()
        print('retriev job(s)  took %.1f sec'%(t2-t1))
        
        postprocess_job_results(rawBitstr,self.meta,self.expD,rawShots)

#...!...!..................
    def postprocRawExperiment(self):        
        raw_experiment_postproc(self.meta,self.expD,self.rawCounter,self.verb)
        
#...!...!..................
    def energySpectrum(self):
        states_energy(self)
       
 
#...!...!..................
    def mathematicalSolution(self):
        empty_123

#...!...!..................
    def fit2ptCorr(self,maxSite=6):
        twoPtCorr=self.expD['magnet_2pt_corr']
        #print('dd',twoPtCorr.shape)
        fmd,Xf,Yf=fit_exponent_simple(twoPtCorr[:maxSite])
        #print(Xf,Yf)

        # save fit results ...
        self.meta['2pt_corr_fit']=fmd
        XYf=np.stack((Xf,Yf), axis=0)
        #print('zz',Xf.shape, XYf.shape)
        self.expD['2pt_corr_fit']=XYf
      
       
        
#...!...!..................
    def twoPointCorrelation(self):
        #... construct data per shot as 2D rectangle array
        amd=self.meta['postproc']
        nAtom=amd['num_atom']
        uShots=amd['used_shots']
        nSol=amd['num_sol']

        dataY=self.expD['ranked_counts']    # [NB]
        hexpattV= self.expD['ranked_hexpatt']
        assert nSol==dataY.shape[0]
        dataS=np.zeros( (uShots,nAtom))

        k=0 # shot index
        for i in range(nSol):
            hexpatt=hexpattV[i]
            mshot=dataY[i]
            A=BitArray(hex=hexpatt)[-nAtom:]  # clip leading 0s
            bitA=[int(x) for x in A.bin]
            #print('hexp:',hexpatt,bitA,mshot)
            for i in range(mshot):
                dataS[i+k]=bitA                
            k+=mshot

        dens_rydberg=np.sum(dataS,axis=0)/uShots
        #print('dataS:%s,\ndensRydberg:'%str(dataS.shape), dens_rydberg)

        if 0:
            #.... compute 2D correlations , the slow way
            densCorr=np.zeros( (nAtom,nAtom))
            for i in range(nAtom):
                for j in range(nAtom):
                    densCorr[i,j]=dens_integrand(i,j,dataS)  # slow, vectorize it

        #.... compute 2D correlations , vectorized
        densCorr=np.cov(dataS.T)

        #.... compute magnetization
        magnetV=do_magnetization(dataS)
        dens_magnet=np.sum(magnetV,axis=0)

        avg_magnet = np.mean(magnetV, axis=0)
        std_magnet = np.std(magnetV, axis=0)
        #1print('avr magnet=',avg_magnet,'\n',std_magnet, magnetV.shape)
        # Compute the density correlation matrix using np.cov with counts as weights
        #1print('rel mag error:',std_magnet/avg_magnet)
        magnetCorr=np.cov(magnetV.T)

        #.... calculate two-point connected correlation for magnetization
        # Set the number of bins, skip 0 (auto-correlation)
        binX=[i for i in range(1,nAtom)]
        twoPtCorr=[]
        for disp in binX:
            corr=connected_two_point_corr_func_v2(nAtom, disp, magnetCorr)
            #print('disp=%d  corrFn=%.3g'%(disp,corr))
            twoPtCorr.append(corr)

        #.... two-point correlation for absolute value of the density-density 
        # Set the number of bins, skip 0 (auto-correlation)
        dtwoPtCorr=[]
        for disp in binX:
            corr=connected_two_point_corr_func_v2(nAtom, disp, np.abs( densCorr))
            #print('disp=%d  dcorrFn=%.3g'%(disp,corr))
            dtwoPtCorr.append(corr)

        self.expD['dens_rydberg']=dens_rydberg
        self.expD['dens_magnet']=dens_magnet
        self.expD['dens_corr']=densCorr
        self.expD['magnet_corr']=magnetCorr
        # Construct array [N, 2]
        self.expD['magnet_2pt_corr']=np.column_stack((binX, twoPtCorr))  # X,Y
        self.expD['absDens_2pt_corr']=np.column_stack((binX, dtwoPtCorr))  # X,Y


        
#...!...!....................
    def _replicateClucters(self,register):
        pd=self.meta['payload']        
        area=pd['aquila_area']
        M1=1000000
        separation=pd['atom_dist_um']/M1 # in (m)

        pos=register_to_numpy(register)
        print('_RC:pp',pos[-1],type(pos[-1][0]))
        maxX=Decimal(pos[-1][0])
        offsetX=area['width']-maxX
        print('maxX:',maxX)
        if 0:
            maxX=separation*(atoms_l-1)
            maxY=separation*(atoms_w-1)
            print('maxX:',maxX,'maxY:',maxY)
        
        
            offsetY=area['width']-maxY
            print('offset X:',offsetX,type(offsetX))
        ncx=1;ncy=1
        if offsetX> maxX+Decimal('0.000020'): ncx=2  # separation between clusters
        #if offsetY> maxX+Decimal('0.000020'): ncy=2

        atoms = AtomArrangement()
        #... grab x/y of input cluster
        xL=register.coordinate_list(0)
        yL=register.coordinate_list(1)
        #print('xL',type(xL),xL)
        
        nClust=0
        for ix in range(ncx): #... replicate along X
            for iy in range(ncy): #... replicate along Y                
                for x,y in zip(xL,yL):
                    x=x+offsetX*ix
                    y=y+offsetY*iy
                    print(nClust,'xy ',x,y,'ix:',ix,'iy:',iy)
                    atoms.add([x,y])
                nClust+=1
        nAtom=len(xL) # num atoms in theinitial graph
        print('replicateClucters --> %d clust of %d atoms'%(nClust,nAtom))
        pd['num_clust']=nClust
        pd['num_atom_in_clust']=nAtom
        return atoms


#............................
#............................  end of class
#............................

#- - - - - - Auxiliary functions  - - - - - - - - -

#...!...!....................
def generate_line(nAtom: int, md,dist=4.0*1e-6):
    # ......  Aquila FOV ......
    fov=md['payload']['aquila_conf']['area']
    fovY=fov['height']
    
    G = nx.Graph()
    atoms = AtomArrangement()
    y0=fovY*Decimal('0.8')
    for i in range(0,nAtom):
        xy=[ i*dist, y0]  # G does not like Decimal attribute
        atoms.add(xy)
        G.add_node(i) #, pos=(x, y))
        if i>0: G.add_edge(i-1, i)
    
    #print('create G nodes:',G.nodes)
    #print('G pos:',pos)
               
    print('line graph has %d nodes'%nAtom)
    return atoms, G


#...!...!....................
def generate_uturn_line(nAtom: int,md, dist, xoffset=0):
    # ......  Aquila FOV ......
    fov=md['payload']['aquila_conf']['area']
    fovX=fov['width']
    fovY=fov['height']
    spaceY=md['payload']['aquila_conf']['geometry']['spacingVerticalMin']
    #print('tt',type(fovX),fovX,spaceY,dist)
    assert dist>spaceY+Decimal('0.2e-6') # to make tringle solvable??
    
    def place_line(N,x0,y0,dist):
        for i in range(N):
            x=i*dist+x0
            k=G.number_of_nodes()#len(pos) # node id k=len(pos) # node id
            xy=[x,y0]
            atoms.add(xy)
            G.add_node(k)
        return x+dist
    
    def place_uturn(x0,y0):
        for i in range(6):
            if i<3:
                x=x0+i*dx ; y=y0+i*spaceY
            else:
                x=x0+dx*(5-i) ;  y=y0+(i-1)*spaceY+dist             
            xy=[x,y]
            k=G.number_of_nodes()
            atoms.add(xy)
            G.add_node(k)
        return x-dist,y

    G = nx.Graph()
    atoms = AtomArrangement()
    nArc=6
    nRowT=(nAtom-nArc)//2
    assert nRowT>0
    nRowB=nAtom-nRowT-nArc
    M=10000  # digital accuracy
    dx=np.sqrt( float(dist)**2- float(spaceY)**2 )
    dx=Decimal('%.8f'%dx)
    #print('uturn check: dx/um',dx*1000000,'dist/um',dist*1000000)

    # place bottom row
    x0=xoffset; y0=Decimal('20e-6')
    x0=place_line(nRowB,x0,y0,dist)
    
    #...  place uturn
    x0,y0=place_uturn(x0,y0)

    # place top row
    x0=place_line(nRowT,x0,y0,-dist)
  
    for i in range(1,nAtom):
        G.add_edge(i-1, i)
    
    #print('create G nodes:',G.nodes,'Rarch:',Rarch)   
    
    print('uturn has %d nodes:'%nAtom)
    return atoms, G


#...!...!....................
def generate_closed_loop(nAtom: int,md,ctype):

    nvert=3;  nhors=3; nhorl=11; m=1
    assert nAtom in [13, 17, 33, 39, 47, 50, 58] # the only viable option 
    if nAtom==50:
        nhorl=10
        m=0
        
    M1=1000000
    # ......  Aquila FOV ......
    fov=md['payload']['aquila_conf']['area']
    fovX=fov['width']
    fovY=fov['height']
    spaceY=md['payload']['aquila_conf']['geometry']['spacingVerticalMin']
    
    dist=md['payload']['atom_dist_um']/M1 # in (m)
    tilt=dist*Decimal('0.7071')  # sin(45 deg)
    assert dist>spaceY
    #print('ddd',dist, 6.2e-6)
    assert dist<=6.21e-6  # to not exceed FOV
    
    def place_vert(N,x0,y0,sn=1):
        for i in range(N):
            y=sn*i*dist+y0
            atoms.add([x0,y])
        return y
    def place_horz(N,x0,y0,sn=1):
        for i in range(N):
            x=sn*i*dist+x0
            atoms.add([x,y0])
        return x
        
    G = nx.Graph()
    atoms = AtomArrangement()
    
       
    M=10000  # digital accuracy
    x0=0; y0=tilt
    #... left climb up ....
    y0=place_vert(nvert+m,x0,y0)
    x0+=tilt; y0+=tilt
    x0=place_horz(nhors,x0,y0)#Y
    xb=x0; yb=y0 # to break loop symmetry
    x0+=tilt; y0+=tilt
    y0=place_vert(nvert,x0,y0)
    x0-=tilt; y0+=tilt
    x0=place_horz(nhors,x0,y0, sn=-1)
    if nAtom>13:
        x0-=tilt; y0+=tilt
        y0=place_vert(nvert+m,x0,y0)

    #..... top snake ...
    x0+=tilt; y0+=tilt
    if nAtom>17:
        x0=place_horz(nhorl,x0,y0)
        # ... right fall down ...   
        x0+=tilt; y0-=tilt
        y0=place_vert(nvert+m,x0,y0, sn=-1)
        x0-=tilt; y0-=tilt
 
    if nAtom==33:  atoms.add([x0,y0])
    if nAtom>33:
        x0=place_horz(nhors+m,x0,y0, sn=-1)
        x0-=tilt; y0-=tilt
        y0=place_vert(nvert,x0,y0, sn=-1)
    if nAtom>39:
        x0+=tilt; y0-=tilt
        x0=place_horz(nhors+m,x0,y0)
        x0+=tilt; y0-=tilt
        y0=place_vert(nvert+m,x0,y0, sn=-1)
    
    #.....  bottom flat ....
    x0-=tilt; y0-=tilt
    if nAtom>47:
        x0=place_horz(nhorl,x0,y0,sn=-1)
        # .... break symmetry of the loop  - not needed    
        #x0=xb+tilt; y0=yb-tilt  # add to (6)
        #x0=xb+dist+tilt; y0=yb  # add to (7)
        #atoms.add([x0,y0])
       
    G.add_node(0)      
    for i in range(1,nAtom):
        G.add_node(i)
        G.add_edge(i-1, i)
        
    print('loop has %d nodes:'%nAtom)
    return atoms, G




#...!...!....................


#...!...!....................
