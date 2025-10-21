#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Objective: run small  lists of different circuits with different initial states
the args.numSample is the selector for type circuits - it is a hack
in meta-data it is converted to 'tag'

-i 1:  1 circuit, CNOT gate
-i 2:  4 circuits , truth table for Ralph CNOT
-i 3:  4 circuits , truth table for Knill CNOT
-i 4:  1 circuit , BellState with Ralph CNOT
-i 5:  1 circuit , BellState with Knill CNOT
-i 6:  1 circuit , 3q GHZ  with Knill CNOT

runs locally  or on cloud (needs creds)

Records meta-data containing  job_id 
HD5 arrays contain input and output
Use sampler and manual transpiler
Dependence:  qiskit 1.2

Use case: 
./submit_zoo_job.py  --numSample 10 --numShot 2000 --backend  ideal

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

from submit_scan_job import commandline_parser, buildPayloadMeta, harvest_submitMeta
from toolbox.Util_Quandela  import fockState_to_bitStr, bitStr_to_dualRailState, decode_sampler_result,  monitor_async_job

#...!...!....................
def buildZooMeta(args):  # minor modiffications
    md=buildPayloadMeta(args)
    pmd=md['payload']
    pmd['tag']=pmd.pop('num_sample')
    if 'SLOS' not in md['submit']['backend']: assert args.numShot> 1e7 
    
    return md

#...!...!....................
def build_task_cnotTruth(md,noise_source):
    pmd=md['payload']
    sbm=md['submit']
    
    # create CNOT Gate as a processor
    if pmd['tag']==3:
        cnot = pcvl.catalog["heralded cnot"].build_processor()
        pmd['comment']='heralded CNOT truth table, duty fact 2/27'
    else:
        cnot = pcvl.catalog['postprocessed cnot'].build_processor()
        pmd['comment']='end-CNOT truth table, duty fact 1/9'
    minPhoton=2
    pcvl.pdisplay(cnot, recursive=True)

    bitStrL=['00','01','10','11']
    if pmd['tag']==1: bitStrL=['10']
    nCirc=len(bitStrL)
    
    nMode=4
    
    # pack each circuit as independent procesor
    taskL=[None]*nCirc
    for j in range(nCirc):
        bitStr=bitStrL[j]
        fockStt=bitStr_to_dualRailState(bitStr )
        print('proc:%d input bitStr:%s  fockStt:%s'%(j,bitStr,fockStt))
        if sbm['run_local']:
            assert 'SLOS' in sbm['backend']
            proc = pcvl.Processor("SLOS",nMode,noise_source)
        else:
            proc = pcvl.RemoteProcessor(sbm['backend'])
        proc.min_detected_photons_filter(minPhoton)
        proc.add(0, cnot)
        proc.with_input( fockStt)
        #pcvl.pdisplay(proc)
                    
        taskL[j]=proc

    #.... update MD
    sbm['num_circ']=nCirc
    pmd['num_mode']=nMode
    pmd['num_qubit']=2
    pmd['init_bitStr']=bitStrL
    return taskL

#...!...!....................
def build_task_GHZstate(md,noise_source):
    pmd=md['payload']
    sbm=md['submit']

    nCirc=1
    num_qubit=2
    if pmd['tag'] ==6:  num_qubit=3
    # create  CNOT Gate as a processor
    cnotP = pcvl.catalog['postprocessed cnot'].build_processor()
    cnotH = pcvl.catalog["heralded cnot"].build_processor()
    if pmd['tag']==4:        
        pmd['comment']='bellState with postproc-CNOT, (aka Ralph)'
        cnot=cnotP
    if pmd['tag'] in [5,6]:        
        # Aubaert:  each heralded cnots brings two added photons for the heralds.
        # Aubaert:   For Ascella, the cloud indicates that the maximum number of photons is 6. use cnotH+cnotP to circumvent it. See more ideas in  toys/noisy_ghz.py 
        minPhoton=num_qubit
        pmd['comment']='bellState with heralded-CNOT  (aka Knill)'
        cnot=cnotH
        if pmd['tag'] ==6:
            pmd['comment']='3q GHZ-state cnotH+cnotP'
            
    minPhoton=num_qubit
    nMode=2*num_qubit
    bitStr='0'*num_qubit
    fockStt=bitStr_to_dualRailState(bitStr )
    print('BTB a:',bitStr,fockStt,minPhoton,sbm['backend'])
    if sbm['run_local']:
        assert 'SLOS' in sbm['backend']
        proc = pcvl.Processor("SLOS",nMode,noise_source)
    else:
        name=sbm['backend']
        print('backN:',name)
        proc = pcvl.RemoteProcessor(name,m=nMode)

    proc.min_detected_photons_filter(minPhoton)
    proc.add(0, pcvl.BS.H())
    proc.add(0, cnot)
    if pmd['tag'] ==6:  proc.add(2, cnotP)
    proc.with_input( fockStt)
    #pcvl.pdisplay(proc)
                    
    taskL=[proc]

    #.... update MD
    sbm['num_circ']=len(taskL)
    pmd['num_mode']=nMode
    pmd['num_qubit']=num_qubit
    pmd['init_bitStr']=[bitStr]
    return taskL

#...!...!....................
def harvest_results(sampResL,md,bigD):  # many circuits
    pmd=md['payload']
    sbm=md['submit']
    
    nCirc=sbm['num_circ']   
    dutyV=np.zeros(nCirc) 
    outV=[0]*nCirc
    for ic in range(nCirc):
        resD=sampResL[ic]
        outCnt,perf,bitStrL=decode_sampler_result(resD,pmd['num_qubit'],verb=0)
        dutyV[ic]=perf
        outV[ic]=outCnt
    outV=np.array(outV)
    md['postproc']['fin_bitStr']=bitStrL
    
    print('\nHRR comment:',pmd['comment'])
    print('HRR requested shots=%d   backend=%s'%(sbm['num_shot'],sbm['backend']))
    print('HRR fin state:',bitStrL)
    print('HRR table:\n',outV)
    
    print('HRR duty factor:')
    [ print('init state:%s   performance: %.1e'%(pmd['init_bitStr'][i],dutyV[i]))  for i in range(nCirc) ]
    
    bigD['meas']=outV
    bigD['duty_fact']=dutyV
   

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=commandline_parser(shots=100_000)
    np.set_printoptions(precision=5)

    expMD=buildZooMeta(args)
  
    # define noise level for local simu
    noise_model =None
    if expMD['submit']['backend']=='noisy:SLOS':
        noise_model = pcvl.NoiseModel(transmittance=0.05, indistinguishability=0.92, g2=0.03)
   
    if expMD['payload']['tag']<=3:  taskL=build_task_cnotTruth(expMD,noise_model)
    if expMD['payload']['tag'] in [4,5,6]:  taskL=build_task_GHZstate(expMD,noise_model)

    pprint(expMD)
    expD={}
    
    print('.... FIRTS CIRCUIT ..............')
    pcvl.pdisplay(taskL[0])
    
    # ------  construct sampler(.) job ------
    nCirc=expMD['submit']['num_circ']
    jobIdL=[0]*nCirc
    runLocal=expMD['submit']['run_local']
    if runLocal :
        outPath=os.path.join(args.basePath,'meas')
    else:
        outPath=os.path.join(args.basePath,'jobs')
        expMD[ 'submit']['job_ids']=jobIdL
        backendN=expMD[ 'submit']['backend']
        print('M: use cloude service: %s ...'%backendN)
                               
    #.... common processor config
    print('M: write to outPath:',outPath)
    assert os.path.exists(outPath)
                        
    if not args.executeCircuit:
        pprint(expMD)
        print('\nNO execution of circuit, use -E to execute the job\n')
        exit(0)       

    ## -------- bind the data to parametrized circuit & submit -------
    print('   --expName   %s   '%(expMD['short_name'] ))
    
    T0=time()
    for ic in range(nCirc):
        proc=taskL[ic]
        #if ic==0: pcvl.pdisplay(proc)
        sampler = Sampler(proc, max_shots_per_call=args.numShot)
        sampler.default_job_name = '%s_c%d'%(expMD['short_name'],ic)
        
        if runLocal:
            resD=sampler.sample_count()     
            jobIdL[ic]=resD
        else:
            # WARN: submits job to cloud backend
            job = sampler.sample_count.execute_async(args.numShot)  
            print(ic,job.id) 
            jobIdL[ic]=job.id
            
            if ic<nCirc-1 or 1: #  Cannot create more than 1 job(s) with Explorer Offer
                print('circ=%d  back=%s wait for results ...'%(ic,backendN))
                monitor_async_job(job)
                resD=job.get_results()
                if 1:  # debug
                    print('perf:',resD['physical_perf'],resD['logical_perf'])
                    print('counts:',resD['results'])

    harvest_submitMeta(expMD,args,T0)    
    if args.verb>1: pprint(expMD)
    
    if runLocal:
        harvest_results(jobIdL,expMD,expD)
        print('M: got results')
        #...... WRITE  MEAS OUTPUT .........
        outF=os.path.join(outPath,expMD['short_name']+'.meas.h5')
        write4_data_hdf5(expD,outF,expMD)        
        #NO print('   ./postproc_zoo.py  --expName   %s   -p a b   -Y\n'%(expMD['short_name']))
    else:
        #...... WRITE  SUBMIT OUTPUT .........
        outF=os.path.join(outPath,expMD['short_name']+'.ibm.h5')
        write4_data_hdf5(expD,outF,expMD)
        
        print('   ./retrieve_zoo_job.py --expName   %s   \n'%(expMD['short_name'] ))
        #pprint(expMD)


    
