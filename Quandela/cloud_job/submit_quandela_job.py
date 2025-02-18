#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
runs localy  or on cloud (needd creds)

Records meta-data containing  job_id 
HD5 arrays contain input and output
Use sampler and manual transpiler
Dependence:  qiskit 1.2


Use case: 
XXXXX
./submit_ibmq_job.py -E  --numQubits 3 3 --numSample 15 --numShot 8000  --backend   ibm_brussels  


'''
import sys,os,hashlib
import numpy as np
from pprint import pprint
from time import time, localtime

from toolbox.Util_IOfunc import dateT2Str
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
#from toolbox.Util_QiskitV2 import  circ_depth_aziz, harvest_circ_transpMeta

#sys.path.append(os.path.abspath("/qcrank_light"))
#from datacircuits.ParametricQCrankV2 import  ParametricQCrankV2 as QCrankV2, qcrank_reco_from_yields
# Quandeal
import perceval as pcvl
from perceval.components.unitary_components import  BS,PS 
from perceval.algorithm import Sampler


import argparse
#...!...!..................
def commandline_parser(backName="ideal",provName="local sim"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--basePath",default='out',help="head dir for set of experiments")
    parser.add_argument("--expName",  default=None,help='(optional) replaces QuandelajobID assigned during submission by users choice')
 
    # ....Circuit speciffic 
    parser.add_argument('-i','--numSample', default=2, type=int, help='num of XX packed in to the job')

    # .... job running
    parser.add_argument('-n','--numShot',type=int,default=2000, help="shots per circuit")
    parser.add_argument('-b','--backend',default=backName, help="tasks")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")
     
    '''there are 3 types of backend
    a) ideal local proc=pcvl.Processor("Naive", mzi)
    b) remote noisy simu: procN='sim:ascella'          
    c) remote QPU: procN='qpu:ascella'
        proc = pcvl.RemoteProcessor(procN) 
    '''

    args = parser.parse_args()    
    args.provider=provName
    if 'ascella' in args.backend:
        args.provider='Quandela_cloud'

    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
    return args

#...!...!....................
def buildPayloadMeta(args):
    pd={}  # payload
    
    #pd['nq_addr'],pd['nq_data']=args.numQubits
    #pd['num_addr']=1<<pd['nq_addr']
    pd['num_sample']=args.numSample
    #pd['num_qubit']=pd['nq_addr']+pd['nq_data']
    #pd['seq_len']=pd['nq_data']*pd['num_addr']
    
    sbm={}
    sbm['num_shots']=args.numShot
    pom={}
    trm={}

    md={ 'payload':pd, 'submit':sbm ,'transpile':trm, 'postproc':pom}
    
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    md['hash']=myHN

    if args.expName==None:
        if 'ideal' in args.backend: name='ideal_'
        #name='nquera_'  if args.noisySimu else 'ideal_'
        md['short_name']=name+myHN
    else:
        md['short_name']=args.expName

    if args.verb>1:  print('\nBMD:');pprint(md)
    return md


#...!...!....................
def harvest_sampler_submitMeta(job,md,args):
    sd=md['submit']
    sd['job_id']=job.job_id()
    backN=args.backend
    sd['backend']=backN     #  job.backend().name  V2
    
    t1=localtime()
    sd['date']=dateT2Str(t1)
    sd['unix_time']=int(time())
    sd['provider']=args.provider
    print('bbb',args.backend,args.expName)
    if args.expName==None:
        # the  6 chars in job id , as handy job identiffier
        md['hash']=sd['job_id'].replace('-','')[3:9] # those are still visible on the IBMQ-web
        tag=args.backend.split('_')[0]
        md['short_name']='%s_%s'%(tag,md['hash'])
    else:
        myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
        md['hash']=myHN
        md['short_name']=args.expName


#...!...!....................
def harvest_sampler_results(jobL,md,bigD,T0=None):  # many circuits
    pmd=md['payload']
    qa={}

    
    jobRes=job.result()
    #counts=jobRes[0].data.c.get_counts()
    
    if T0!=None:  # when run locally
        elaT=time()-T0
        print(' job done, elaT=%.1f min'%(elaT/60.))
        qa['running_duration']=elaT
    else:
        jobMetr=job.metrics()
        #print('HSR:jobMetr:',jobMetr)
        qa['timestamp_running']=jobMetr['timestamps']['running']
        qa['quantum_seconds']=jobMetr['usage']['quantum_seconds']
        qa['all_circ_executions']=jobMetr['executions']
        
        if jobMetr['num_circuits']>0:
            qa['one_circ_depth']=jobMetr['circuit_depths'][0]
        else:
            qa['one_circ_depth']=None
    
    #1pprint(jobRes[0])
    nCirc=len(jobRes)  # number of circuit in the job
    jstat=str(job.status())
    
    countsL=[ jobRes[i].data.c.get_counts() for i in range(nCirc) ]

    # collect job performance info
    res0cl=jobRes[0].data.c
    qa['status']=jstat
    qa['num_circ']=nCirc
    qa['shots']=res0cl.num_shots
    
    qa['num_clbits']=res0cl.num_bits
    
    print('job QA'); pprint(qa)
    md['job_qa']=qa
    bigD['rec_udata'], bigD['rec_udata_err'] =  qcrank_reco_from_yields(countsL,pmd['nq_addr'],pmd['nq_data'])

    return bigD

#...!...!....................
def build_mzi_param_circ(md): # Mach-Zehnder Interferometers=, parametrized
    # parameterised phase shifter, \phi,  between two fixed beamsplitters.
    mzi = pcvl.Circuit(2) // BS() // (1,PS(phi=pcvl.P("phi"))) // BS()
    pmd=md['payload']
    pmd['num_mode']=2
    return mzi

#...!...!....................
def construct_mzi_inputs(md,verb=1):
    pmd=md['payload']
    # generate float random data
    data_inp = np.random.uniform(0.2, 1.1, size=(pmd['num_sample']))
    if verb>2:
        print('input data=',data_inp.shape)
    bigD={'inp_data': data_inp}
 
    return bigD

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=commandline_parser()
    np.set_printoptions(precision=3)

    expMD=buildPayloadMeta(args)
    pprint(expMD)    
    expD=construct_mzi_inputs(expMD)

    #.... build parametric circuit
    qcP=build_mzi_param_circ(expMD)
    print('.... PARAMETRIZED IDEAL CIRCUIT ..............')
    pcvl.pdisplay(qcP)
    
    # ------  construct sampler(.) job ------
    runLocal=True  # ideal or fake backend
    outPath=os.path.join(args.basePath,'meas') 
    if 'ideal' in args.backend:
        proc=pcvl.Processor("Naive",qcP)
    else:
        print('M: activate QiskitRuntimeService() ...')
        service = QiskitRuntimeService()
        if  'fake' in args.backend:
            transBackN=args.backend.replace('fake_','ibm_')
            hw_backend = service.backend(transBackN)
            backend = AerSimulator.from_backend(hw_backend) # overwrite ideal-backend
            print('fake noisy backend =', backend.name)
        else:
            outPath=os.path.join(args.basePath,'jobs')
            assert 'ibm' in args.backend
            backend = service.backend(args.backend)  # overwrite ideal-backend
            print('use true HW backend =', backend.name)          
            runLocal=False
            outPath=os.path.join(args.basePath,'jobs')
        qcT =  transpile(qcP, backend,optimization_level=3)
        qcrankObj.circuit=qcT  # pass transpiled parametric circuit back
        cxDepth=qcT.depth(filter_function=lambda x: x.operation.name == 'cz')
        print('.... PARAMETRIZED Transpiled (%s) CIRCUIT .............., cx-depth=%d'%(backend.name,cxDepth))
        print('M: transpiled gates count:', qcT.count_ops())
        if args.verb>2 or nq_addr<4:  print(qcT.draw('text', idle_wires=False))
                
    #.... common processor config
    inpML=[1,0] # initial Fock state
    proc.min_detected_photons_filter(1)
    input_state = pcvl.BasicState(inpML)
    proc.with_input(input_state)

    sampler = Sampler(proc, max_shots_per_call=args.numShot)    
    #1harvest_circ_transpMeta(qcT,expMD,backend.name)
    assert os.path.exists(outPath)

    nCirc=expMD[ 'payload']['num_sample']
    print('M: execution-ready %d circuits  backend=%s'%(nCirc,args.backend))
                            
    if not args.executeCircuit:
        pprint(expMD)
        print('\nNO execution of circuit, use -E to execute the job\n')
        exit(0)
        

    # -------- bind the data to parametrized circuit & submit -------
    T0=time()
    jobIdL=[0]*nCirc
    for ic in range(nCirc):
        theta=expD['inp_data'][ic]
        qcP.get_parameters()[0].set_value(theta)
        sampler.default_job_name = '%s_c%d'%(expMD['short_name'],ic)

        job = sampler.sample_count.execute_async(args.numShot)  # Create a job
        if runLocal:
            jobIdL[ic]=job
        else:
            print(ic,job.id) 
            jobIdL[ic]=job.id

    '''
    monitor_job(job)
    y=abs(mzi.compute_unitary()[0,0])**2  
    Yt[i]=y
    '''
    
    #1harvest_sampler_submitMeta(job,expMD,args)    
    if args.verb>1: pprint(expMD)
    
    if runLocal:
        harvest_sampler_results(jobIdL,expMD,expD,T0=T0)
        print('M: got results')
        #...... WRITE  MEAS OUTPUT .........
        outF=os.path.join(outPath,expMD['short_name']+'.meas.h5')
        write4_data_hdf5(expD,outF,expMD)        
        print('   ./postproc_qcrank.py  --expName   %s   -p a    -Y\n'%(expMD['short_name']))
    else:
        #...... WRITE  SUBMIT OUTPUT .........
        outF=os.path.join(outPath,expMD['short_name']+'.ibm.h5')
        write4_data_hdf5(expD,outF,expMD)
        print('M:end --expName   %s   %s  %s  jid=%s'%(expMD['short_name'],expMD['hash'],backend.name ,expMD['submit']['job_id']))
        print('   ./retrieve_ibmq_job.py --expName   %s   \n'%(expMD['short_name'] ))



    
