__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import time
from pprint import pprint
import numpy as np
import qiskit as qk
from toolbox.UQiskit_job import harvest_submitInfo, retrieve_qiskit_job, harvest_retrievInfo, postprocess_job_results

import qiskit.qasm3

#............................
#............................
#............................
class TaskTomoTeleport():
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
       
        # transpiler conf
        tpc={'optimization_level':args.transpOptLevel}  

        #if args.provider=='IBMQ':
        tpc.update({'seed_transpiler':args.randomSeed,'initial_layout':args.qubits})  
        # spare scheduling_method='alap'
  
        csd={} # circuits  summary
        csd['phys_qubits']=args.qubits
        csd['text_qubits']= '+'.join([str(x) for x in args.qubits ])
        csd['barrier']=not args.executeCircuit  # drop all barriers for full execution of the circuit, do I need it?
                
        # submit info
        smd={'num_shots':args.numShots, 'transp_conf':tpc, 'provider':args.provider}
        smd['backend']=args.backendName
        #smd['ideal_simu']= args.backendName=='ibmq_qasm_simulator'

        # analyzis info
        amd={'ana_code':'ana_tomoTele.py'}

        pd={}  # payload
        pd['class_name']=self.__class__.__name__
        pd['theta'],pd['phi']=args.initState
        pd['info']='ini theta:%.1f phi:%.1f q:%s'%(pd['theta'],pd['phi'],csd['text_qubits'])
        
        md={'circ_sum':csd , 'payload':pd,'submit':smd,'analyzis':amd}
        md['short_name']=args.expName
            
        self.meta=md
        if self.verb>1:
            print('BMD:');pprint(md)
            
#...!...!....................
    def buildCirc(self):
        qcL=[]
        # teleportation circ w/ 3 tomo-projections
        for  axMeas in list('zyx'):
            qcTele=self._circTeleport() # Alic-Bob teleportation circuit
            self._meas_tomo(qcTele,2,2,axMeas)
            if self.verb>1:   print('BC',qcTele.name); print(qcTele)
            qcL.append(qcTele)
        
        csd=self.meta['circ_sum']
        csd['num_circ']=len(qcL)
        csd['num_qubits']=qcTele.num_qubits
        csd['num_clbits']=csd['num_qubits']
        csd['num_mbasis']=1<<csd['num_clbits']  # number of measured bitstrings
        csd['circ_name']=[ qc.name for qc in qcL ] # may be used in analysis
        csd['num_mid_clbits']=2  # mid-circ measurement
        self.circL=qcL
        
#...!...!....................
    def _circTeleport(self):
        csd=self.meta['circ_sum']
        pd=self.meta['payload']
        name='tele_q'+csd['text_qubits']
        assert len(csd['phys_qubits'])==3
        
        ## SETUP
        # Protocol uses 3 qubits and 2 classical bits in 2 different registers
        # include third register for measuring Bob's result
        qr = qk.QuantumRegister(3, name="q")
        crz, crx = qk.ClassicalRegister(1, name="mz_alice"), qk.ClassicalRegister(1, name="mx_alice") # Alice
        crb = qk.ClassicalRegister(1, name="bob")  # Bob
        qc = qk.QuantumCircuit(qr, crz, crx, crb,name=name)

        qs=0;qa=1;qb=2  # soft ids of qubits: secrte, a+b form Bell state

        # Creates secret quantum initial state
        qc.u(pd['theta'],pd['phi'],0.0,0) # theta,phi,lambda
        qc.barrier()

        # Creates a bell pair in qc using qubits a & b
        qc.h(qa) # Put qubit a into state |+>
        qc.cx(qa,qb) # CNOT with a as control and b as target
        qc.barrier()

        # Alice measures secret state in Z & X basis using one of bell-qubits
        self._alice_send(qc, qs, qa, crz, crx)

        # Bob receives 2 classical bits and applies gates on bell-state b-qubit
        self._bob_gates(qc, qb, crz, crx)

        return qc
#...!...!....................
    def _alice_send(self,qc, qs, qa, crz, crx):
        """Measures qubits a & b and 'sends' the results to Bob"""
        qc.cx(qs, qa)
        qc.h(qs)
        qc.measure(qs,crz)
        qc.measure(qa,crx)
        qc.barrier()

#...!...!....................
    # This function takes a QuantumCircuit (qc), integer (qubit)
    # and ClassicalRegisters (crz & crx) to decide which gates to apply
    def _bob_gates(self,qc, qubit, crz, crx):
        # Here we use qc.if_test to control our gates with a classical
        # bit instead of a qubit
        with qc.if_test((crx, 1)):
            qc.x(qubit)
        with qc.if_test((crz, 1)):
            qc.z(qubit)
           
#...!...!....................
    def _meas_tomo(self,qc,tq,tb,axMeas):
        assert axMeas in 'xyz'
        if axMeas=='y':
            qc.sx(tq)

        if axMeas=='x':
            qc.rz(np.pi/2,tq)  # it should have been -pi/2
            qc.sx(tq)
            qc.rz(-np.pi/2,tq) # it should have been +pi/2

        qc.measure(tq,tb)
        qc.name+='_tomo_'+axMeas


#...!...!....................
    def transpileIBM(self,backend):       
        tcf=self.meta['submit']['transp_conf']
        
        t4=time.time()
        qcTL = qk.transpile(self.circL, backend, **tcf)
        t5=time.time()
        print('transpiled6, took %.1f sec, conf:'%(t5-t4),tcf)

        self.circTL=qcTL        
        self._export_qasm3()

#...!...!....................
    def _export_qasm3(self):
        ncirc=len(self.circL)
        #... orig circ
        self.expD['circ_qasm3']=np.empty((ncirc), dtype='object')
        for i,qc in enumerate(self.circL):
            self.expD['circ_qasm3'][i]=qiskit.qasm3.dumps( qc )
            
        #... transpiled circ
        self.expD['transp_qasm3']=np.empty((ncirc), dtype='object')
        for i,qc in enumerate(self.circTL):
            self.expD['transp_qasm3'][i]=qiskit.qasm3.dumps( qc )
            

#...!...!..................
    def postprocessIBM_submit(self,job):        
        harvest_submitInfo(job,self.meta,taskName='midcm')
        

#...!...!..................
    def retrieveIBM_job(self,provider,job=None):
        
        isBatch= 'batch_handles' in self.expD  # my flag
        
        if job==None:
            smd=self.meta['submit']
            jid=smd['job_id']        
            job = retrieve_qiskit_job(provider, jid, verb=self.verb)
            
        t1=time.time()    
        harvest_retrievInfo(job.result(),self.meta)
        t2=time.time()
        print('got harvest_retrievInfo, took %.1f sec'%(t2-t1))

        if isBatch:
            never_tested22
            jidL=[x.decode("utf-8") for x in expD['batch_handles']]
            print('jjj',jidL)
            jobL = [backend.retrieve_job(jid) for jid in jidL ]
            resultL =[ job.result() for job in jobL]
            jobMD['submit']['cost']*=len(jidL)
        else:
            rawCountsL=job.result().get_counts()
            if type(rawCountsL)!=type([]): rawCountsL=[rawCountsL] # Qiskit is inconsitent, requires this patch - I want always a list of results

        
        postprocess_job_results(rawCountsL,self.meta,self.expD)
