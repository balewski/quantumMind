__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from pprint import pprint
import time
import os, hashlib

import numpy as np
import qiskit as qk
from qiskit_ibm_provider import IBMProvider

# for remap QQQ
from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import SetLayout, ApplyLayout
from bitstring import BitArray
import argparse

'''  So many options to choos from:
Provider:
IBMQ (cloud): ideal-sim, noisy-sim, real-HW
Aer  (local): ideal-sim, noisy-sim, fake-HW
'''

#...!...!..................
def submit_args_parser(backName,parser=None):
    if parser==None:  parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or env--> $QCloud_dataVault")
    parser.add_argument("--expName",  default=None,help='(optional) replaces IBMQ jobID assigned during submission by users choice')
    
    #parser.add_argument("-M", "--mathOps",  default='',choices=['msbflip', '1compl','midthresh','2rowsum','2rowgrad','hammw'], nargs='+',help="list of color operations, order matters, space separated")   

   
    parser.add_argument('-q','--qubits', default=[2,3], nargs='+', type=int, help='physical qubit ids, space separated')
    
    
    # Qiskit:
    parser.add_argument('-b','--backendName',default=backName,help="backend for computations " )
    parser.add_argument('--provider',default='aer',choices=['ibmq','aer'],help="compute provider " )
    parser.add_argument('-n','--numShots', default=1000, type=int, help='num of shots')
    parser.add_argument('-r','--randomSeed',type=int,default=111, help="transpiler seed ")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")
     
    parser.add_argument('-L','--transpOptLevel',type=int,default=1, help="transpiler optimization level ")
    parser.add_argument('--qasmDump',type=int,default=0, help="0: none, 1: one non-parametric transpiled circ, 2: undefined")

    if 0 :# hack1
        if backName=='noisy_sim':
            parser.add_argument("-N", "--noiseModel",  default='minor',choices=['minor','ibmq','qtuum'],help="select pre-canned magnitude of noise")
    
    args = parser.parse_args()
    if 'env'==args.basePath: args.basePath= os.environ['QCloud_dataVault']
    args.outPath=os.path.join(args.basePath,'jobs')
        
    if args.backendName=='ideal':  args.backendName='ibmq_qasm_simulator'
    
    for arg in vars(args):
        if '_' in arg: continue # skip canned input
        print( 'myArgs:',arg, getattr(args, arg))

    assert os.path.exists(args.outPath)
    return args

#...!...!....................
def access_ibm_provider( provider,verb=1):
    print('ask for IBMProvider()...')
    t1=time.time()
    provider = IBMProvider()
    t2=time.time()
    print('M: got provider, took %.1f sec'%(t2-t1))

    if 0:
        from qiskit_ibm_provider import least_busy
        backend1 = least_busy(provider.backends())
        print("M: least_busy backend:", backend1.name)
        
    if verb>1:
        qasm_backends = set( backend.name for backend in provider.backends())
        print("M:The following backends are accessible:", qasm_backends)
        #assert args.backendName in qasm_backends
        
    return provider

#...!...!....................
def access_qiskit_backend( provider, backName,verb=1):
    
    if provider=='ibmq':
        assert 'ibm' in backName
        provider= access_ibm_provider( provider,verb=verb)
        backend = provider.get_backend(backName)
        
    #...........
    if provider=='aer':
        assert 'ibm' not in backName
        if backName=='aer_simulator':
            backend = qk.Aer.get_backend(backName)
        if 'fake' in  backName :
            fakeBack=access_fake_backend(backName)
            from qiskit.providers.aer import AerSimulator
            backend = AerSimulator.from_backend(fakeBack)
            #print('bnnn',backend.backend_name); ok99
            #print('back',backend,'num_qubits:',backend.configuration().num_qubits); ok6
    return backend

#...!...!....................
def access_fake_backend(backName, verb=1): 
    import importlib
    assert 'fake' in backName
    a,b=backName.split('_')
    b0=b[0].upper()
    backName='Fake'+b0+b[1:] #+'V2'
    print('zzz',a,b,backName)
    module = importlib.import_module("qiskit.providers.fake_provider")
    cls = getattr(module, backName)
    return cls()

#...!...!.................... 
def harvest_submitInfo(job,md,taskName='exp'):
    sd=md['submit']
    sd['job_id']=job.job_id()

    backend=job.backend()
    
    t1=time.localtime()
    sd['date']=dateT2Str(t1)
    sd['unix_time']=int(time.time())
    

    if md['short_name']==None:
        # the  6 chars in job id , as handy job identiffier
        md['hash']=sd['job_id'].replace('-','')[:7] # those are still visible on the IBMQ-web
        name=taskName+'_'+md['hash']
        #?if md['submit']['ideal_simu']: name='is'+name
        #1if 'ionq.qpu' in  md['backend']['name']: name='ionexp_'+md['hash']
        md['short_name']=name
    else:
        myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
        md['hash']=myHN
        
        

#...!...!....................
def retrieve_qiskit_job(provider, job_id, verb=1):
    from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus
    print('\nretrieve_job provider=',provider,' search for',job_id)
    job=None
    try:
        job = provider.retrieve_job(job_id)
    except:
        print('job NOT found, quit\n')
        exit(99)

    print('job IS found, retrieving it ...')
    startT = time.time()
    job_status = job.status()
    while job_status not in JOB_FINAL_STATES:
        print('Status  %s , queue  position: %s,    mytime =%d ' %(job_status.name,str(job.queue_position()),time.time()-startT))
        #print(type(job_status)); pprint(job_status)
        time.sleep(30)
        job_status = job.status()

    print('final job status:',job_status)
    if job_status!=JobStatus.DONE :
        print('abnormal job termination, job_id=',job_id)
        # ``status()`` will simply return``JobStatus.ERROR`` and you can call ``error_message()`` to get more
        print(job.error_message())
        exit(0)
        
    return  job 


#...!...!.................... 
def harvest_retrievInfo(results,md):
    # working w/ dict gives the most flexibility
    headD = results.to_dict()  # it probably recomputes all bits strings as well
    # collect job performance info    
    qaD={}
    for x in ['success',  'status','job_id']:  
        qaD[x]=headD[x]
    qaD['exec_date']=str(headD['date'])
    
    print('job QA'); pprint(qaD)
    md['job_qa']=qaD
    
    md['submit']['info']='exec %s,'%qaD['exec_date'][5:16]



#...!...!..................
def postprocess_job_results(rawCountsL,md,expD,backend=None):
    if type(rawCountsL)!=type([]): rawCountsL=[rawCountsL] # Qiskit is inconsitent, requires this patch - I want always a list of results

    csd=md['circ_sum']
    counts_raw=np.array(0)  # empty stub, replaced by 1st circut data (*M)
    
    for ic in range(csd['num_circ']):
        if csd['num_mid_clbits']>0 :  # pairs of measurements
            cntV=unpack_qiskit_midCM_counts(rawCountsL[ic])
        else:
            cntV=unpack_qiskit_counts(rawCountsL[ic],csd['num_clbits'])
        print('circ:%s\n'%csd['circ_name'][ic],cntV)
        if counts_raw.ndim==0:
            counts_raw=np.zeros( (csd['num_circ'],)+cntV.shape, dtype=cntV.dtype)
            #print('create counts_raw:',counts_raw.shape)
                
        counts_raw[ic]=cntV
    expD['counts_raw']=counts_raw
    #print('ee',counts_raw)
 
#...!...!....................
def unpack_qiskit_midCM_counts(qk_counts,toBigEndian=False):
    # convert qiskit.result.counts.Counts to numpy [ncirc, prem,postm]
    # Qiskit is using LSB order, aka small-endian

    keyL=list(qk_counts)
    key0=keyL[0]
    #print('mm',key0, 'NB post:',postNB, 'pre:',preNB)
    print('cc',qk_counts)
    bitsL=key0.split(' ')
    print('aa bitsL',bitsL)
    dimV=tuple()
    for bb in bitsL:
        dimV+=(1<<len(bb),)
    #print('dimV',bb,dimV)
    counts=np.zeros(dimV,dtype=np.int32)
    if len(keyL)==0:  return counts 
    
  
    if toBigEndian:  # not efficient - make it better if you know how
        not_impl

    for key in keyL:
        bitsL=key.split(' ')

        adrV=tuple()
        for bb in bitsL:
            valA=BitArray(bin=bb)
            adrV+=(valA.uint,)
        #print('bb',bb,adrV)
        counts[adrV]=qk_counts[key]  # retain Qiksit order
    
    return counts

#...!...!....................
def unpack_qiskit_counts(qk_counts,post_clbits,toBigEndian=False):
    # convert qiskit.result.counts.Counts to numpy [ncirc, prem,postm]
    # Qiskit is using LSB order, aka small-endian

    postNB=1<<post_clbits # number of possible bitstrings
    counts=np.zeros(postNB,dtype=np.int32)
    
    keyL=list(qk_counts)
    key0=keyL[0]
    print('mm',key0, 'NB post:',postNB)
    print('cc',qk_counts)
    if len(keyL)==0:  return counts 
    
    # verify number of measured qubits
    postS=key0        
    assert len(postS)==post_clbits

    if toBigEndian:  # not efficient - make it better if you know how
        not_impl2

    for key in keyL:
        postA=BitArray(bin=key)
        counts[postA.uint]=qk_counts[key]  # retain Qiksit order
    
    return counts


''' - - - - - - - - -
time-zone aware time, usage:

*) get current date:
t1=time.localtime()   <type 'time.struct_time'>

*) convert to string:
timeStr=dateT2Str(t1)

*) revert to struct_time
t2=dateStr2T(timeStr)

*) compute difference in sec:
t3=time.localtime()
delT=time.mktime(t3) - time.mktime(t1)
delSec=delT.total_seconds()
or delT is already in seconds
'''

#...!...!..................
def dateT2Str(xT):  # --> string
    nowStr=time.strftime("%Y%m%d_%H%M%S_%Z",xT)
    return nowStr

#...!...!..................
def dateStr2T(xS):  #  --> datetime
    yT = time.strptime(xS,"%Y%m%d_%H%M%S_%Z")
    return yT

   
# --- OLD code ----

#...!...!....................
def QQQremap_qubits(qc,targetMap):
    # input: new target order of the current qubit IDs
    # access quantum register
    qr=qc.qregs[0]
    nq=len(qr)
    assert len(targetMap)==nq
    #print('registers has %d qubist'%nq,qr)
    regMap={}
    for i,j in enumerate(targetMap):
        #print(i,'do',j)
        regMap[qr[j]]=i

    #print('remap qubits:'); print(regMap)
    layout = Layout(regMap)
    #print('layout:'); print(layout)

    # Create the PassManager that will allow remapping of certain qubits
    pass_manager = PassManager()

    # Create the passes that will remap your circuit to the layout specified above
    set_layout = SetLayout(layout)
    apply_layout = ApplyLayout()

    # Add passes to the PassManager. (order matters, set_layout should be appended first)
    pass_manager.append(set_layout)
    pass_manager.append(apply_layout)

    # Execute the passes on your circuit
    remapped_circ = pass_manager.run(qc)

    return remapped_circ


