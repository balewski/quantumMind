#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Objective:  run parametrized circuit as function of paramater

runs localy  or on cloud (needa creds)

Records meta-data containing  job_id 
HD5 arrays contain input and output
Use sampler and manual transpiler
Dependence:  qiskit 1.2

Use case: 
./submit_scan_job.py  --numSample 10 --numShot 2000 --backend ideal

'''
import sys,os,hashlib
import numpy as np
from pprint import pprint
from time import time, localtime

from toolbox.Util_IOfunc import dateT2Str
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5


# Quandela
import perceval as pcvl
from perceval.components.unitary_components import  BS,PS 
from perceval.algorithm import Sampler
from toolbox.Util_Quandela  import  monitor_async_job

import argparse
#...!...!..................
def commandline_parser(backName="ideal",provName="local sim",shots=20_000):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--basePath",default='out',help="head dir for set of experiments")
    parser.add_argument("--expName",  default=None,help='(optional) replaces QuandelajobID assigned during submission by users choice')
 
    # ....Circuit speciffic 
    parser.add_argument('-i','--numSample', default=3, type=int, help='num of XX packed in to the job')

    # .... job running
    parser.add_argument('-n','--numShot',type=int,default=shots, help="shots per circuit")
    parser.add_argument('-b','--backend',default=backName, choices=['ideal','noisy','twin','qpu'],help="tasks")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")
     
    '''there are 3 types of backend
    a,b) ideal local proc=pcvl.Processor("SLOS")  or noisy_local
    c,d) remote digital twin: procN='sim:ascella'          
           remote QPU: procN='qpu:ascella'
        proc = pcvl.RemoteProcessor(procN) 
    '''

    args = parser.parse_args()    
    args.provider=provName
    if 'ideal' not in args.backend:
        args.provider='Quandela cloud'
        
    print('perceval ver:',pcvl.__version__)
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
    return args

#...!...!....................
def buildPayloadMeta(args):
    pd={}  # payload
    pd['num_sample']=args.numSample
    
    sbm={}
    sbm['num_shot']=args.numShot
    if 'ideal' in args.backend: backendN='ideal:SLOS'
    elif 'noisy' in args.backend: backendN='noisy:SLOS'
    elif 'twin' in args.backend: backendN='sim:ascella'
    elif 'qpu' in args.backend: backendN='qpu:ascella'
    else:
        print('BPM unknown backend%s requested, ABORT'%args.backend); exit(99)

    sbm['backend']=backendN
    sbm['perceval_ver']=pcvl.__version__
    sbm['run_local']='SLOS' in backendN
    
    pom={}
    trm={}

    md={ 'payload':pd, 'submit':sbm ,'transpile':trm, 'postproc':pom,'qa':{}}
    
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    md['hash']=myHN

    if args.expName==None:
        md['short_name']='%s_%s'%(args.backend,myHN)
    else:
        md['short_name']=args.expName

    if args.verb>1:  print('\nBMD:');pprint(md)
    return md


#...!...!....................
def harvest_submitMeta(md,args,T0=None):
    sbm=md['submit']
    qam=md['qa']
    t1=localtime()
    sbm['date']=dateT2Str(t1)
    sbm['unix_time']=int(time())
    sbm['provider']=args.provider
    #print('bbb',args.backend,args.expName)
    if sbm['run_local']:
        elaT=time()-T0
        print(' job done, elaT=%.1f min'%(elaT/60.))
        qam['running_duration']=elaT

#...!...!....................
def harvest_sampler_results(jobL,md,bigD):  # many circuits
    pmd=md['payload']
    
    Xt=bigD['inp_data']
    Yt=bigD['truth']
    nCirc=len(jobL)
    print('HSL nCirc=%d'%(nCirc))
    Ym=np.zeros((nCirc,2)) # [circ] (val,err)
    perfV=np.zeros(nCirc)
    for ic in range(nCirc):
        resD=jobL[ic] # tmp, ideal simu
        perf=resD['physical_perf']
        cntC=resD['results']
        n0,n1=0,0
        for k,v in cntC.items():
            #print(k,v)
            if k==pcvl.BasicState('|1,0>'): n1=v
            if k==pcvl.BasicState('|0,1>'): n0=v
        ns=n0+n1
        prob=n1/(ns)
        probEr=np.sqrt(prob*(1-prob)/ns) if n0*n1>0 else 1/ns
        Ym[ic]=[prob,probEr]
        perfV[ic]=perf
        print('ic=%d  transm=%.2g  n0=%d n1=%d  prob=%.3f +/- %.3f  Yt=%.3f  theta=%.1f rad'%(ic, perf,n0,n1,prob,probEr,Yt[ic],Xt[ic]))
    avrPerf=np.mean(perfV)
    print('M:OK avrPerf=%.2g'%avrPerf)
        
   
    
    '''
    
    if T0!=None:  # when run locally
       
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
    '''
    
    bigD['rec_data']=Ym
    bigD['transmitance'] = perfV

    return bigD

#...!...!....................
def construct_circ_and_data(md):    # it is a pair 
    #.... build parametric circuit
    qcP=build_mzi_param_circ(expMD)
    # input data
    expD=construct_mzi_inputs(expMD)
    return qcP, expD
    
#...!...!....................
def build_mzi_param_circ(md): # Mach-Zehnder Interferometer, parametrized
    # parameterised phase shifter, \phi,  between two fixed beam splitters.
    mzi = pcvl.Circuit(2) // BS() // (1,PS(phi=pcvl.P("phi"))) // BS()
    pmd=md['payload']
    pmd['num_mode']=2
    return mzi

#...!...!....................
def construct_mzi_inputs(md,verb=1):
    pmd=md['payload']
    # generate float random data
    #data_inp = np.linspace(0.2, 1.7, pmd['num_sample'])
    data_inp = np.linspace(-0.4, 3.6, pmd['num_sample'])  # 10pt hits 0 & pi
    if verb>0: print('input data=',data_inp.shape,data_inp)
   
    Yt=(1-np.cos(data_inp))/2
    bigD={'inp_data': data_inp,'truth':Yt}
 
    return bigD


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=commandline_parser()
    np.set_printoptions(precision=3)
   
    if 0:  # backup plan
        token= os.getenv('MY_QUANDELA_TOKEN')
        print('perceval ver:',pcvl.__version__)
        pcvl.save_token(token)
        print(token)
        
    expMD=buildPayloadMeta(args)
    pprint(expMD)

    qcP, expD= construct_circ_and_data(expMD)

    # define noise level
    if expMD['submit']['backend']=='noisy:SLOS':
        noise_gen = pcvl.Source(emission_probability=0.25, multiphoton_component=0.01)
    else:
        noise_gen=None
        
    print('.... PARAMETRIZED IDEAL CIRCUIT ..............')
    pcvl.pdisplay(qcP)
    
    # ------  construct sampler(.) job ------
    nCirc=expMD[ 'payload']['num_sample']
    nMode=expMD[ 'payload']['num_mode']
    jobIdL=[0]*nCirc
    runLocal=expMD['submit']['run_local']

    if runLocal:        
        outPath=os.path.join(args.basePath,'meas')
        proc = pcvl.Processor("SLOS",nMode,noise_gen)
    else:
        expMD[ 'submit']['job_ids']=jobIdL
        backendN=expMD[ 'submit']['backend']
        print('M: use cloude service: %s ...'%backendN)
        proc = pcvl.RemoteProcessor(backendN,m=nMode)          
        outPath=os.path.join(args.basePath,'jobs')
                       
    #.... common processor config
    assert os.path.exists(outPath)
    proc.set_circuit(qcP)
    proc.min_detected_photons_filter(1)
    input_state = pcvl.BasicState([1,0])  # quSt='0', dual rail encoding
    proc.with_input(input_state)
    sampler = Sampler(proc, max_shots_per_call=args.numShot)    
    print('M: execution-ready %d circuits  backend=%s'%(nCirc,args.backend))
                            
    if not args.executeCircuit:
        pprint(expMD)
        print('\nNO execution of circuit, use -E to execute the job\n')
        exit(0)       

    # -------- bind the data to parametrized circuit & submit -------
    print('   --expName   %s   '%(expMD['short_name'] ))

    T0=time()
    for ic in range(nCirc):
        theta=expD['inp_data'][ic]
        qcP.get_parameters()[0].set_value(theta)
        sampler.default_job_name = '%s_c%d'%(expMD['short_name'],ic)
        pcvl.pdisplay(proc)
     
        if runLocal:
            resD=sampler.sample_count()               
            jobIdL[ic]=resD
        else:
            # WARN: submits job to cloud backend
            job = sampler.sample_count.execute_async(args.numShot)  
            print(ic,job.id) 
            jobIdL[ic]=job.id
            
            if 1: #  Cannot create more than 1 job(s) with Explorer Offer
                print('wait for results ...',backendN)
                monitor_async_job(job)             
    
    harvest_submitMeta(expMD,args,T0)    
    if args.verb>1: pprint(expMD)
    
    if runLocal:
        harvest_sampler_results(jobIdL,expMD,expD)
        print('M: got results')
        #...... WRITE  MEAS OUTPUT .........
        outF=os.path.join(outPath,expMD['short_name']+'.meas.h5')
        write4_data_hdf5(expD,outF,expMD)        
        print('   ./postproc_scan.py  --expName   %s   -p a b   -Y\n'%(expMD['short_name']))
    else:
        #...... WRITE  SUBMIT OUTPUT .........
        outF=os.path.join(outPath,expMD['short_name']+'.ibm.h5')
        write4_data_hdf5(expD,outF,expMD)
        
        print('   ./retrieve_scan_job.py --expName   %s   \n'%(expMD['short_name'] ))
        #pprint(expMD)


    
