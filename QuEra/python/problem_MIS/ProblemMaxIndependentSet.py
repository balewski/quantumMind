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
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField
from braket.ahs.hamiltonian import Hamiltonian
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation

from toolbox.UAwsQuEra_job import harvest_submitInfo , retrieve_aws_job, harvest_retrievInfo, postprocess_job_results
from toolbox.Util_ahs import compute_nominal_rb, register_to_numpy, drive_to_numpy, register_from_numpy, drive_from_numpy, ramp_drive_waveform, mid_drive_waveform, raw_experiment_analysis, states_energy 
from toolbox.Util_miscIO import  graph_from_JSON, graph_to_JSON


#............................
#............................
#............................
class ProblemMIS():  # Maximal-Independent-Set
#...!...!....................
    def __init__(self, args, jobMD=None, expD=None):
        self.verb=args.verb
        print('Task:',self.__class__.__name__)
        if jobMD==None:
            self._buildSubmitMeta(args)
            self.expD={}
        else:
            assert jobMD['payload']['class_name']==self.__class__.__name__
            self.meta=jobMD
            self.expD=expD
            #self._body_from_bigData()
            self._ahs_problem_from_bigData()    
           
    @property
    def submitMeta(self):
        return self.meta
    @property
    def numCirc(self):
        return len(self.circL)
    def getShortName(self):
        return self.submitMeta['short_name']
    def setShortName(self,txt):
        self.submitMeta['short_name']=txt
    def atoms_pos_um(self):
        return self.expD['atoms_xy']*1e6
    def getDrive(self):
        times,amplitudes,detunings,phases=self.expD['hamilt_drive']
        return times,amplitudes,detunings,phases
    
#...!...!....................
    def _ahs_problem_from_bigData(self):
        bigD=self.expD
        self.register=register_from_numpy(bigD['atoms_xy'])
        self.H=drive_from_numpy(bigD['hamilt_drive'])
        self.graph=graph_from_JSON(bigD['graph.JSON'][0])
        if 'counts_raw.JSON' in bigD:
            # -------  counts  -------
            rawBitstr=json.loads(bigD['counts_raw.JSON'][0])
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
    def _replicateClucters(self,register):
        pd=self.meta['payload']
        atoms_l, atoms_w=pd['grid_shape']
        if atoms_l=='square': atoms_l=atoms_w  # hack
        atoms_l=int(atoms_l)
        atoms_w=int(atoms_w)
        
        area=pd['aquila_area']
        M1=1000000
        separation=pd['atom_dist_um']/M1 # in (m)
        maxX=separation*(atoms_l-1)
        maxY=separation*(atoms_w-1)
        print('maxX:',maxX,'maxY:',maxY)
        
        offsetX=area['width']-maxX
        offsetY=area['width']-maxY
        print('offset X:',offsetX,type(offsetX))
        ncx=1;ncy=1
        if offsetX> maxX+Decimal('0.000020'): ncx=2  # separation between clusters
        if offsetY> maxX+Decimal('0.000020'): ncy=2

        atoms = AtomArrangement()
        #... grab x/y of input cluster
        xL=register.coordinate_list(0)
        yL=register.coordinate_list(1)
        #print('xL',type(xL),xL)
        
        nClust=0
        for ix in range(ncx): #... replicate along X
            for iy in range(ncy): #... replicate along Y                
                for x,y in zip(xL,yL):
                    if ix==1: x=x+offsetX
                    if iy==1: y=y+offsetY
                    print(nClust,'xy ',x,y,'ix:',ix,'iy:',iy)
                    atoms.add([x,y])
                nClust+=1
        nAtom=len(xL) # num atoms in theinitial graph
        print('replicateClucters --> %d clust of %d atoms'%(nClust,nAtom))
        pd['num_clust']=nClust
        pd['num_atom_in_clust']=nAtom
        return atoms
    
#...!...!....................
    def _buildSubmitMeta(self,args):
      
        smd={'num_shots':args.numShots}
        smd['backend']=args.backendName
        
        # analyzis info
        amd={'ana_code':'ana_MIS.py'}

        pd={}  # payload
        pd['atom_dist_um']=Decimal(args.atom_dist_um)
        pd['grid_shape']=args.grid_shape
        pd['grid_seed']=args.grid_seed
        pd['grid_droput']=args.grid_droput
        pd['evol_time_us']=args.evol_time_us
        pd['rabi_ramp_time_us']=args.rabi_ramp_time_us
        pd['detune_shape']=args.detune_shape
        pd['rabi_omega_MHz']=args.rabi_omega_MHz               
        pd['class_name']=self.__class__.__name__
        pd['num_clust']=args.multi_clust #tmp assignment, see placeAtoms(.)
        md={ 'payload':pd,'submit':smd,'analyzis':amd} 
        md['short_name']=args.expName            

        if pd['grid_seed']>0 : np.random.seed(pd['grid_seed'])
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
        pd['aquila_area']=AQUILA['area']

        M1=1000000
        separation=pd['atom_dist_um']/M1 # in (m)
        atoms_l, atoms_w=pd['grid_shape']
        #print('zz',atoms_l,'square',atoms_l=='square')
        if atoms_l=='square':  # special case
            register, G = generate_square_circle(int(atoms_w),  scale=separation)
        else:
            register, G = generate_unit_disk(int(atoms_l), int(atoms_w), scale=separation,dropout=pd['grid_droput'])
                    
            
        if pd['num_clust']==False:
            pd['num_clust']=1
            pd['num_atom_in_clust']=len(register)
        else:            
            register=self._replicateClucters(register)
        
        self.register=register
        self.graph=G
        nAtom=len(register)
        
        #... update meta-data
        
        pd['num_atom']=nAtom
        pd['info']='atoms:%d,omega:%.1f,dist:%.1f,seed:%d'%(nAtom,pd['rabi_omega_MHz'],pd['atom_dist_um'],pd['grid_seed'])
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
        delta_begin = -13 *MHz2pi
        delta_end =  11 * MHz2pi
        omega_max = pd['rabi_omega_MHz'] * MHz2pi # units rad/sec      

        t_max  =pd['evol_time_us']/M1
        t_up =pd['rabi_ramp_time_us'][0]/M1
        t_down =pd['rabi_ramp_time_us'][1]/M1
        t_vary=t_max - t_up - t_down
        assert t_vary>0.1e-6 
        Ha=ramp_drive_waveform(t_up,omega_max,delta_begin,'pre')        
        Hb=mid_drive_waveform(t_vary,omega_max,delta_begin,delta_end,pd['detune_shape'] )
        Hc=ramp_drive_waveform(t_down,omega_max,delta_end,'post')
        H=Ha.stitch(Hb).stitch(Hc)
        if self.verb>1:  print('BH:',drive_to_numpy(H))
        self.H=H
        pd['nominal_Rb_um']=compute_nominal_rb(omega_max, 0.)*1e6

        """Stitches two driving fields based on TimeSeries.stitch method.
        The time points of the second DrivingField are shifted such that the first time point of
        the second DrifingField coincides with the last time point of the first DrivingField.
        The boundary point value is handled according to StitchBoundaryCondition argument value.
        """




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
        harvest_submitInfo(job,self.meta,taskName='mis')
                

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
         
        t2=time.time()
        print('retriev job(s)  took %.1f sec'%(t2-t1))
        
        postprocess_job_results(rawBitstr,self.meta,self.expD)

#...!...!..................
    def analyzeRawExperiment(self):
        raw_experiment_analysis(self.meta,self.expD,self.rawCounter,self.verb)
          
#...!...!..................
    def energySpectrum(self):
        states_energy(self)

       
 
#...!...!..................
    def mathematicalSolution(self):
        # 1. A maximal independent set is an independent set that cannot be enlarged by adding any other vertex from the graph.
        # 2. Maximal Independent Sets (MIS) can have different cardinalities because they are not required to be of the same size in a given graph.
        
        G=self.graph        
        complG = nx.complement(G)
        self.complGraph=complG
        all_misL = bron_kerbosch(complG)
        nNode=G.number_of_nodes()
        
        #.... order MISs for easier assesement ....
        # Step 1: Convert sets to lists
        lists_list = [sorted(s) for s in all_misL]
        # Step 2: Sort the lists by length in descending order
        sorted_lists = sorted(lists_list, key=len, reverse=True)
        maxCard=len(sorted_lists[0])
        minCard=len(sorted_lists[-1])
        nMIS=len( lists_list)
        
        if 1: # Step 3: Print the sorted list of lists            
            print("All %d Maximal Independent Sets:"%nMIS)
            for lst in sorted_lists:
                print(lst)

        # pack it for HD5
        IC=2 # Id & cardinality
        nMx=0; nMi=0
        assert nNode<=64 # see below 'dtype'
        outV=np.zeros((nMIS,IC),dtype=np.uint64)  # MIS-ID & carinality 
        for i in range(nMIS):
            mis=sorted_lists[i]
            card=len(mis)
            if card==maxCard: nMx+=1
            if card==minCard: nMi+=1
            A= BitArray(length=nNode)
            A.set(1, mis)
            outV[i]= A.uint,card
            

        # pack  results in several ways
        tmd={}
        self.meta['true_math']=tmd
        tmd['max_card']=maxCard
        tmd['min_card']=minCard
        tmd['num_MISs_any_card']=nMIS
        tmd['num_graph_nodes']=nNode
        tmd['num_graph_edges']=G.number_of_edges()
        tmd['num_MISs_max_card']=nMx
        tmd['num_MISs_min_card']=nMi
        
        self.expD['true_MISs_any_card']=outV
        self.expD['true_MISs_max_card']=outV[:nMx,0] # no need to keep cardinality
        self.expD['true_MISs_min_card']=outV[-nMi:,0]
        pprint(tmd)
        #pprint(outV)
        #pprint( self.expD['true_MISs_max_card'])
 
        if self.verb<=1: return

        # Set 4: Group lists by their lengths
        result = {}; maxCard=-1
        for sublist in sorted_lists:
            length = len(sublist)
            if maxCard< length : maxCard=length
            if length in result:
                result[length].append(sublist)
            else:
                result[length] = [sublist]

        # Print the separate lists according to length
        for length, sublists in sorted(result.items()):
            print(f"Length {length}:")
            for sublist in sublists:
                print(sublist)
            print()  
        
     

#............................
#............................  end of class
#............................

#- - - - - - Auxiliary functions  - - - - - - - - -
        
#...!...!....................
def generate_unit_disk(atoms_l: int, atoms_w: int, scale=4.0*1e-6, dropout=0.45):
    # https://en.wikipedia.org/wiki/Unit_distance_graph
    # atoms_l,w : length and width of the atom grid
    atom_list = []
    edge_dict = {}
    atom_to_edge = {}
    kk=-1
    for ii in range(atoms_l):
        for jj in range(atoms_w):
            kk+=1  # absolute grid point ID
            atom_list.append((ii*scale, jj*scale,kk))
            atom_to_edge[(ii*scale, jj*scale)] = ii*atoms_w + jj
            edge_dict[ii*atoms_w + jj] = []
            if jj < atoms_w - 1:
                edge_dict[ii*atoms_w + jj].append(ii*atoms_w + jj + 1)
            if ii < atoms_l - 1:
                edge_dict[ii*atoms_w + jj].append((ii+1)*atoms_w + jj)
            if ii < atoms_l - 1 and jj < atoms_w - 1:
                # nearest neighbor
                edge_dict[ii*atoms_w + jj].append((ii+1)*atoms_w + jj + 1)
            if jj > 0 and ii < atoms_l - 1:
                # nearest neighbor
                edge_dict[ii*atoms_w + jj].append((ii+1)*atoms_w + jj - 1)

    graph = nx.from_dict_of_lists(edge_dict)
    
    # perform dropout
    new_len = int(np.round(len(atom_list) * (1 - dropout)))
    atom_arr = np.empty(len(atom_list), dtype=object)
    atom_arr[:] = atom_list
    remaining_atom_list = np.random.choice(atom_arr, new_len, replace=False)
    
    # order atoms according to node ID
    nAtom=len( remaining_atom_list)
    idL=[ i for i in range(nAtom)]  # to be reversed list of atoms
    for i in range(nAtom):
        myid=remaining_atom_list[i][2]
        idL[i]=myid
    #print('idL:',idL)
      
    # Construct a list of indices that will sort the list
    indices = sorted(range(len(idL)), key=lambda x: idL[x])
    #print('atom indices:',indices)
    # Define a mapping of old node IDs to new node IDs
    node_mapping = {idL[indices[i]]:i for i in range(nAtom)}
    #print("node_mapping:",node_mapping)
   
    atoms = AtomArrangement()
    for j in indices:
        atom=remaining_atom_list[j][:2]
        atoms.add(atom)
           
    graph.remove_nodes_from([atom_to_edge[atom[:2]] for atom in  set(atom_list) - set(list(remaining_atom_list))])
    # Rename the nodes of the graph 
    graph = nx.relabel.relabel_nodes(graph, node_mapping)
    print('unit disk graph has %d nodes'%nAtom)
    
    return atoms, graph

#...!...!....................
def generate_square_circle(nAtoms_x: int, scale=4.0*1e-6):
    
    def generate_square_graph(N):
        G = nx.Graph()
        # Add nodes
        for i in range(4 * N):
            G.add_node(i)
        # Add edges
        for i in range(4 * N):
            G.add_edge(i, (i + 1) % (4 * N))
        return G

    def get_square_node_positions(N):
        positions = {}
        for i in range(N):
            positions[i] = (i, 0)
            positions[N + i] = (N, i)
            positions[2 * N + i] = (N - i, N)
            positions[3 * N + i] = (0, N - i)
        return positions

    N = nAtoms_x-1
    G = generate_square_graph(N)
    pos = get_square_node_positions(N)
    nAtom=len(pos)

    print('G nodes:',G.nodes)
    
    atoms = AtomArrangement()
    for ia in range(nAtom):
        ix,iy=pos[ia]
        atoms.add([ ix*scale, iy*scale])
           
    print('square-cric graph has %d nodes'%nAtom)
    return atoms, G


#...!...!....................
def bron_kerbosch(graph, clique=set(), candidates=None, excluded=set()):
    if candidates is None:
        candidates = set(graph.nodes)
    
    if not candidates and not excluded:
        yield clique
        return

    for node in list(candidates):
        new_candidates = candidates.intersection(graph.neighbors(node))
        new_excluded = excluded.intersection(graph.neighbors(node))
        yield from bron_kerbosch(graph, clique | {node}, new_candidates, new_excluded)
        candidates.remove(node)
        excluded.add(node)

