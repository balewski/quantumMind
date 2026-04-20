#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Template for submiting single AHS  (Analog Hamiltonian Simulator) job to QuEra
- optional submision to Aqula (Neutral Atom QPU)
- no graphics

Submits job to AWS-Braket service
Records meta-data containing  task_arn 
HD5 arrays contain:
- 
- ??input data(optional)


Dependence:  Braket

Use cases: 
* 2-atom Ball-state
./submit_awsQuEra_job.py --numShots 8000 --backendName cpu  --num_atom 2 --atom_dist_um 12 --rabi_omega_MHz 1.8 --evol_time_us 3.4

* 3x3 grid covering FOV
./submit_awsQuEra_job.py  --numShots 200   --num_atom 9 --atom_in_column  3 --atom_dist_um 24.8   --rabi_omega_MHz 1.8 --evol_time_us 2.0

* 4x4 grid
./submit_awsQuEra_job.py  --numShots 200   --num_atom 16 --atom_in_column  4 --atom_dist_um 20.   --rabi_omega_MHz 1.8 --evol_time_us 2.0


'''

import sys,os
import time
from pprint import pprint 

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.UAwsQuEra_job import submit_args_parser, access_quera_device
from toolbox.UAwsQuEra_job import  harvest_retrievInfo, postprocess_job_results  # for emulator
# one is needed:
from ProblemAtomGridSystem import ProblemAtomGridSystem 

import argparse

#...!...!..................
def get_parser(backName=None):  # add task-speciffic args
    parser = argparse.ArgumentParser()

    #  ProblemAtomGridSystem task speciffic
    parser.add_argument('--atom_dist_um', default='9.0', type=str, help='distance between 2 atoms, in um')
    parser.add_argument('--detune_MHz', default=0., type=float, help='detuning frequency in MHz')

    parser.add_argument('--num_atom', default=8, type=int, help='number of atoms total')
    parser.add_argument('--atom_in_column', default=3, type=int, help='number of atoms per column')
    parser.add_argument("-g","--gridType",  default='square',choices=["square","hex"],help="abc-string listing shown plots")

                    
    args=submit_args_parser(backName,parser)
    
    return args


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser(backName='cpu')  # qpu

    # choose your task, pick one
    task=ProblemAtomGridSystem(args) 

    task.placeAtoms()
    task.buildHamiltonian()
    ahs_program=task.buildProgram()  # schoredingerProblem

    jobMD=task.submitMeta    
    if args.verb>1 : pprint(jobMD)
    
    if 0 :  # debug when meta-data packing to JSON crahses
        import ruamel.yaml  as yaml
        ymlFd = open('dump.yaml', 'w')
        yaml.dump(jobMD, ymlFd, Dumper=yaml.CDumper)
        ymlFd.close() 
        end_json
    
    shots=args.numShots
    device=access_quera_device(args.backendName,verb=args.verb)
    
    if 0 and args.backendName!='emulator':
        capabilities = device.properties.paradigm
        pprint(capabilities.dict()); ok77

    if  args.backendName=='emulator':  # shortcut for local simulation
        evol_step_us=100
        evolSteps=int(jobMD['payload']['evol_time_us']*evol_step_us)
        print('\n Emulator: evol time %.3f us , steps=%d'%(float(jobMD['payload']['evol_time_us']),evolSteps)) 
        job = device.run(ahs_program, shots=shots, steps=evolSteps)
        task.postprocess_submit(job.metadata())    
        
        harvest_retrievInfo(None,task.meta)

        rawCounts=job.result().get_counts()
        postprocess_job_results(rawCounts,task.meta,task.expD)
        #...... WRITE  TASK  OUTPUT .........
        args.outPath=os.path.join(args.basePath,'meas')
        outF=os.path.join(args.outPath,jobMD['short_name']+'.h5')
        write4_data_hdf5(task.expD,outF,task.meta)
        
        baseStr= " --basePath %s "%args.basePath if 'env'!=os.environ['QuEra_dataVault']  else ""
        anaCode= jobMD['analyzis']['ana_code']
        print('\n   ./%s %s --expName   %s   -X \n'%(anaCode,baseStr,jobMD['short_name']))
        exit(0)
    else:        
        #Before submitting the AHS program to Aquila, we need to discretize the program to ensure that it complies with resolution-specific validation rules.

        discr_ahs_program = ahs_program.discretize(device)
        if 0:
            circD=discr_ahs_program.to_ir().dict()
            pprint(circD);
            circD=ahs_program.to_ir().dict()
            pprint(circD); ok51
        
        
    if not args.executeTask:
        print('NO execution of AHS-program, use -E to execute the job on: %s , shots=%d'%(args.backendName,shots))
        exit(0)    
    
    # ----- Aquila submission ----------
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

    jobMD=task.submitMeta    
           
    if args.verb>0: pprint(jobMD)
    
           
    #...... WRITE  JOB META-DATA .........
    outF=os.path.join(args.outPath,jobMD['short_name']+'.quera.h5')
    write4_data_hdf5(task.expD,outF,jobMD)
    print('M:end --expName   %s   %s  %s  ARN=%s'%(jobMD['short_name'],jobMD['hash'],args.backendName ,jobMD['submit']['task_arn']))
    print('   ./retrieve_awsQuEra_job.py --expName   %s  \n'%(jobMD['short_name'] ))
   
