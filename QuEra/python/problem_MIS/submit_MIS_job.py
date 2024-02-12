#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Solve Max Independent Set  probelm (Analog Hamiltonian Simulator) job to QuEra
- optional submision to Aqula (Neutral Atom QPU)
- no graphics

Submits job to AWS-Braket service
Records meta-data containing  task_arn 
HD5 arrays contain:
- meta-data w/ full QuEra problem
- atoms postion
- probability matrix

Dependence:  Braket

Use cases: 
 4x3 grid with 
 ./submit_MIS_job.py --atom_dist_um 4.1 --evol_time_us 0.6 --rabi_omega_MHz 2.5 -n100


'''

import sys,os
import time
from pprint import pprint
import numpy as np

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.UAwsQuEra_job import submit_args_parser, access_quera_device
from toolbox.UAwsQuEra_job import  harvest_retrievInfo, postprocess_job_results  # for emulator
from ProblemMaxIndependentSet import ProblemMIS

import argparse

#...!...!..................
def get_parser(backName="cpu"):  # add task-speciffic args
    parser = argparse.ArgumentParser()

    #  ProblemMIS task speciffic
    parser.add_argument('--atom_dist_um', default='4.1', type=str, help='distance between 2 atoms, in um')

    parser.add_argument('--grid_shape', default=[3,4],  nargs='+', help='grid x & y size, space separated list, OR:   square & x-size ')
    parser.add_argument('-s','--grid_seed', default=92, type=int, help='seed for graph generator, 0=none')
 
    parser.add_argument('--detune_shape', default=[0.25,0.5,0.75], type=float, nargs='+', help='relative intermediate values of detune, space separated list') 

    parser.add_argument( "-M","--multi_clust",   action='store_true', default=False, help="replicate  cluster multipl times")

    parser.add_argument('--grid_droput', default=0.45, type=float, help='fraction of removed atoms from the perfect grid')


    args=submit_args_parser(backName,parser)
    
    return args


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    # choose your task, pick one
    task=ProblemMIS(args) 
    
    task.placeAtoms()

    task.buildHamiltonian()
    
    ahs_program=task.buildProgram()  # SchrodingerProblem
   
    jobMD=task.submitMeta
    baseStr= " --basePath %s "%args.basePath if args.basePath!=os.environ['QuEra_dataVault']  else ""  # for next step program
    if args.verb>1 : pprint(jobMD)
        
    shots=args.numShots

    if 0 and args.backendName!='emulator':
        capabilities = device.properties.paradigm
        pprint(capabilities.dict()); ok77

    #........ shortcut for local simulation
    if  args.backendName=='emulator' and not args.fakeSubmit:
        device=access_quera_device(args.backendName,verb=args.verb)
        
        evolSteps=int(jobMD['payload']['evol_time_us']*100)
        print('\n Emulator: evol time %.3f us , steps=%d'%(float(jobMD['payload']['evol_time_us']),evolSteps))
        # Set a different seed based on the current time
        np.random.seed(int(time.time()))

        job = device.run(ahs_program, shots=shots, steps=evolSteps, solver_method="bdf")
        task.postprocess_submit(job.metadata())
                    
        harvest_retrievInfo(None,task.meta)

        rawCounts=job.result().get_counts()
        if len(rawCounts)<20:
            print("M:rawCounts:")
            pprint(rawCounts)
        from collections import Counter
        occurence_count = Counter(rawCounts)
        most_frequent_regs = occurence_count.most_common(3)
        print("M:most_frequent"); pprint(most_frequent_regs)
   
        
        postprocess_job_results(rawCounts,task.meta,task.expD)
        #...... WRITE  TASK  OUTPUT .........
        args.outPath=os.path.join(args.basePath,'meas')
        outF=os.path.join(args.outPath,jobMD['short_name']+'.h5')
        write4_data_hdf5(task.expD,outF,task.meta)
        anaCode= jobMD['analyzis']['ana_code']
        print('\n   ./%s --expName  %s  %s   -p r d s e -X\n'%(anaCode,baseStr, jobMD['short_name']))
        exit(0)
    else:        
        #Before submitting the AHS program to Aquila, we need to discretize the program to ensure that it complies with resolution-specific validation rules.

        
        if 0:
            circD=discr_ahs_program.to_ir().dict()
            pprint(circD);
            circD=ahs_program.to_ir().dict()
            pprint(circD); ok51
        
        
   
    if  args.executeTask:    
        # ----- Aquila submission ----------
        device=access_quera_device(args.backendName,verb=args.verb)
        discr_ahs_program = ahs_program.discretize(device)
        job = device.run(discr_ahs_program, shots=shots)  
        jid=job.id
        if 0:
            pprint(job)
            print('job arn:', job.id)
            print('job state:', job.state())
            print('job meta:'); pprint( job.metadata())

        print('submitted JID=',jid)
        task.postprocess_submit(job)    

        if args.cancelTask:
            print('M: cancel job:',job.id)
            job.cancel()

        
    elif args.fakeSubmit:  # special case
        task.postprocess_submit(None)
        task.submitMeta['short_name']='f'+task.submitMeta['short_name']
    else:
        print('NO execution of AHS-program, use -E to execute the job on: %s , shots=%d'%(args.backendName,shots))
        exit(0)    

    jobMD=task.submitMeta       
    if args.verb>0: pprint(jobMD)
    #...... WRITE  JOB META-DATA .........
    outF=os.path.join(args.outPath,jobMD['short_name']+'.quera.h5')
    write4_data_hdf5(task.expD,outF,jobMD)
    print('M:end --expName   %s   %s  %s  ARN=%s'%(jobMD['short_name'],jobMD['hash'],args.backendName ,jobMD['submit']['task_arn']))
    

    if args.fakeSubmit:
            print('   ./math_MIS.py %s --expName   %s  -p r d g m n   -X\n'%(baseStr,jobMD['short_name'] ))
    else:
            print('   ./retrieve_awsQuEra_job.py %s --expName   %s  \n'%(baseStr,jobMD['short_name'] ))

    print('aa', args.basePath,os.environ['QuEra_dataVault'])
