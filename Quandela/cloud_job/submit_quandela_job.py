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
import inspect
#print(inspect.getsource(qiskit_aer.AerSimulator))



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
    parser.add_argument('-b','--backend',default=backName, choices=['ideal','noisy','qpu'],help="tasks")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")
     
    '''there are 3 types of backend
    a) ideal local proc=pcvl.Processor("SLOS", mzi)
    b) remote noisy simu: procN='sim:ascella'          
    c) remote QPU: procN='qpu:ascella'
        proc = pcvl.RemoteProcessor(procN) 
    '''

    args = parser.parse_args()    
    args.provider=provName
    if 'ideal' not in args.backend:
        args.provider='Quandela cloud'

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
    if 'ideal' in args.backend: backendN='ideal:SLOS'
    else:  backendN=args.backend+':ascella'
    sbm['backend']=backendN
    
    pom={}
    trm={}

    md={ 'payload':pd, 'submit':sbm ,'transpile':trm, 'postproc':pom}
    
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    md['hash']=myHN

    if args.expName==None:
        md['short_name']='%s_%s'%(args.backend,myHN)
    else:
        md['short_name']=args.expName

    if args.verb>1:  print('\nBMD:');pprint(md)
    return md


#...!...!....................
def harvest_submitMeta(md,args):
    sbm=md['submit']
    '''
    if 'ideal' in sbm['backend']:
        jid=md['hash']
    else:
        jid=
    sd['job_id']=job.job_id()
    backN=args.backend
    sd['backend']=backN     #  job.backend().name  V2
    '''
    t1=localtime()
    sbm['date']=dateT2Str(t1)
    sbm['unix_time']=int(time())
    sbm['provider']=args.provider
    print('bbb',args.backend,args.expName)
    


#...!...!....................
def harvest_sampler_results(jobL,md,bigD,T0=None):  # many circuits
    pmd=md['payload']
    qa={}

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
    aaa
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
    '''
    bigD['rec_data']=Ym
    bigD['transmitance'] = perfV

    return bigD

#...!...!....................
def construct_circ_and_data(md):

    # it is a pair 
    #.... build parametric circuit
    qcP=build_mzi_param_circ(expMD)
    # input data
    expD=construct_mzi_inputs(expMD)
    return qcP, expD
    
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
    data_inp = np.linspace(0.2, 1.7, pmd['num_sample'])
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

 
    
    if 0:
        token= os.getenv('MY_QUANDELA_TOKEN')
        token='_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTA5NiwiZXhwIjoxNzQyNzAyNDk2fQ.8rhXLqTg7cwdljEDs_qOZqYqFeVslEAUHCMs1z-hfBVHuhN68fQJ3EwNOLgKSeJCqbUtrs7hQtBaMMWPyrdAMQ'
        print('perceval ver:',pcvl.__version__)
        pcvl.save_token(token)
    
    expMD=buildPayloadMeta(args)
    pprint(expMD)

    qcP, expD= construct_circ_and_data(expMD)
   
    print('.... PARAMETRIZED IDEAL CIRCUIT ..............')
    pcvl.pdisplay(qcP)
    
    # ------  construct sampler(.) job ------
    nCirc=expMD[ 'payload']['num_sample']
    jobIdL=[0]*nCirc
    runLocal=True  # ideal or fake backend
    outPath=os.path.join(args.basePath,'meas')
    if 'ideal' in args.backend:
        proc = pcvl.Processor("SLOS")
    else:
        expMD[ 'submit']['job_ids']=jobIdL
        backendN=expMD[ 'submit']['backend']
        backendN='sim:ascella'
        print('M: use cloude service: %s ...'%backendN)
        if 1:
            #pcvl.save_token('_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTA5NiwiZXhwIjoxNzM5OTQ1MDUwfQ.F4kP3s8vc8BPOAsKRKuQtKrPWkU9cIwKFB-3tLNlbrfjwTdREm5HiAJpa39zo_pH54xd_4Cpay9mFSb1h9sx4Q')
            proc = pcvl.RemoteProcessor("sim:ascella")
            print('perceval ver:',pcvl.__version__)
            #proc = pcvl.RemoteProcessor(backendN,qcP)          
        runLocal=False
        outPath=os.path.join(args.basePath,'jobs')
       
                
    #.... common processor config
    assert os.path.exists(outPath)
    proc.set_circuit(qcP)
    inpML=[1,0] # initial Fock state
    proc.min_detected_photons_filter(1)
    input_state = pcvl.BasicState(inpML)
    proc.with_input(input_state)

    sampler = Sampler(proc, max_shots_per_call=args.numShot)    
    #1harvest_circ_transpMeta(qcT,expMD,backend.name)

    print('M: execution-ready %d circuits  backend=%s'%(nCirc,args.backend))
                            
    if not args.executeCircuit:
        pprint(expMD)
        print('\nNO execution of circuit, use -E to execute the job\n')
        exit(0)
        

    # -------- bind the data to parametrized circuit & submit -------
    T0=time()
 
   
    for ic in range(nCirc):
        theta=expD['inp_data'][ic]
        qcP.get_parameters()[0].set_value(theta)
        sampler.default_job_name = '%s_c%d'%(expMD['short_name'],ic)
        pcvl.pdisplay(proc)
        '''
        job = sampler.sample_count.execute_async()  # Create a job
        #print(inspect.getsource(Sampler))
        
        #print(ic,job.id)
        resD = job.get_results()
        perf=resD['physical_perf']
        cntC=resD['results']
        n0,n1=0,0
        for k,v in cntC.items():
            print(k,v)     
            if k==pcvl.BasicState('|1,0>'): n1=v
            if k==pcvl.BasicState('|0,1>'): n0=v
        vvvv
        sample_count = sampler.sample_count(1000)
        print('results:',sample_count['results'])
        print('physical_perf:',sample_count['physical_perf'])
        hhh
        job = sampler.sample_count.execute_async(args.numShot)  # Create a job
        '''
        if runLocal:
            resD=sampler.sample_count()               
            jobIdL[ic]=resD
            #print(ic,'ideal:',jobIdL[ic])
        else:
            
            #1job = sampler.sample_count.execute_async(args.numShot)  # Create the job
            #1print(ic,job.id) 
            #1jobIdL[ic]=job.id
            jobIdL[ic]='041033f9-f535-4e0e-a0b2-267aa8c78bc7'
    
    harvest_submitMeta(expMD,args)    
    if args.verb>1: pprint(expMD)
    
    if runLocal:
        harvest_sampler_results(jobIdL,expMD,expD,T0=T0)
        print('M: got results')
        #...... WRITE  MEAS OUTPUT .........
        outF=os.path.join(outPath,expMD['short_name']+'.meas.h5')
        write4_data_hdf5(expD,outF,expMD)        
        print('   ./postproc_quandela.py  --expName   %s   -p a    -Y\n'%(expMD['short_name']))
    else:
        #...... WRITE  SUBMIT OUTPUT .........
        outF=os.path.join(outPath,expMD['short_name']+'.ibm.h5')
        write4_data_hdf5(expD,outF,expMD)
        #1print('M:end --expName   %s   %s  %s  jid=%s'%(expMD['short_name'],expMD['hash'],backendN ,expMD['submit']['job_id']))
        print('   ./retrieve_quandela_job.py --expName   %s   \n'%(expMD['short_name'] ))
        pprint(expMD)


    
