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
class TaskMidCircMeas():
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
        amd={'ana_code':'ana_midCM.py'}

        pd={}  # payload
        pd['class_name']=self.__class__.__name__
        pd['info']='ini  q: %s '%csd['text_qubits']
 
        md={'circ_sum':csd , 'payload':pd,'submit':smd,'analyzis':amd}
        md['short_name']=args.expName
            
        self.meta=md
        if self.verb>1:
            print('BMD:');pprint(md)
            
#...!...!....................
    def buildCirc(self):
        qcL=[]
        for c in [False,True]:
            qc=self._circMcondM(cond=c)
            if self.verb>1:
                print('BC',qc.name); print(qc)
            qcL.append(qc)
        csd=self.meta['circ_sum']
        csd['num_circ']=len(qcL)
        csd['num_qubits']=qc.num_qubits
        csd['num_clbits']=csd['num_qubits']
        csd['num_mbasis']=1<<csd['num_clbits']  # number of measured bitstrings
        csd['circ_name']=[ qc.name for qc in qcL ] # may be used in analysis
        csd['num_mid_clbits']=1  # mid-circ measurement
        self.circL=qcL

#...!...!....................
    def _circMcondM(self,cond):
        csd=self.meta['circ_sum']
        if cond:
            name='MCM_q'+csd['text_qubits']
        else:
            name='MXM_q'+csd['text_qubits']
        numq=len(csd['phys_qubits'])
        assert numq==2
        
        crPre = qk.ClassicalRegister(1, name="m_pre")
        crPost = qk.ClassicalRegister(numq, name="m_post")
        qr = qk.QuantumRegister(numq, name="q")
        qc= qk.QuantumCircuit(qr, crPre,crPost,name=name)
        qc.h(0)
        qc.x(1)
        qc.measure(0,crPre)
        if cond:
            with qc.if_test((crPre, 0)):
                qc.x(0)
                qc.x(1)
        else:
            qc.x(0)
            qc.barrier()
        qc.measure(0,crPost[0])
        qc.measure(1,crPost[1])
        return qc


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
