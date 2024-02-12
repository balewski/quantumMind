#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Template for mutiple circuits
- transpiled & mapped to qubits manualy before submission
- no SPAM correction 
- no graphics
Submits job to IBMQ
Records meta-data containing  job_id 
HD5 arrays contain:
- initial and transpiled circ as array of strings, exported as qasm3
- input data(optional)


Dependence:  qiskit

Use cases:
./submit_ibmq_job.py  -E  --provider aer -b aer_simulator
./submit_ibmq_job.py  -E  --provider aer -b fake_kolkata
./submit_ibmq_job.py  -E  --provider ibmq -b ibmq_qasm_simulator
./submit_ibmq_job.py  -E  --provider ibmq -b ibmq_kolkata -q 0 1 2


'''

import sys,os
import time
from pprint import pprint

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

from toolbox.UQiskit_job import access_qiskit_backend, submit_args_parser

# one is needed:
from TaskTomoU3      import TaskTomoU3
from TaskMidCircMeas import TaskMidCircMeas
from TaskTomoTeleport import TaskTomoTeleport

import argparse

#...!...!..................
def get_parser(backName=None):  # add task-speciffic args
    parser = argparse.ArgumentParser()

    # TomoU3 or MidCircMeas task speciffic
    parser.add_argument('--initState', default=[2.4,0.6], type=float,  nargs='+', help='initial state defined by U3("theta/rad", "phi/rad", "lam=0") ')
                    
    args=submit_args_parser(backName,parser)
    
    return args


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser(backName='aer_simulator')

    # choose your task, pick one
    #1args.qubits=args.qubits[0:1] ; task=TaskTomoU3(args); 
    #1task=TaskMidCircMeas(args)
    task=TaskTomoTeleport(args) 
  
    task.buildCirc()

    jobMD=task.submitMeta    
    pprint(jobMD)
   
    if not args.executeCircuit:
        print('NO execution of %d circuits, use -E to execute the job on: %s'%(task.numCirc,args.backendName))
        exit(0)

    backend=access_qiskit_backend(args.provider,args.backendName,verb=args.verb)
    

    tot_qubits=backend.configuration().num_qubits
    print('my backend=',backend,' num phys qubits:',tot_qubits)

    if args.verb>1:  print('M: base gates:', backend.configuration().basis_gates)
    
    task.transpileIBM(backend)

    qcTL=task.circTL
    if args.verb>0:
        print('M:....  CIRCUIT TRANSPILED ..............')
        print(qcTL[0].draw(output="text", idle_wires=False))
       
    if args.qasmDump==1:
        fix_me
        nqubit=sum(args.numQubits); npix=codex.meta['payload']['num_pixels']
        qasmF='qbart_%dqub_%dpix.qasm'%(nqubit,npix)
        qasmF=os.path.join(args.outPath,qasmF)
        circEL[0].qasm(filename=qasmF)    # save QASM
        print(circEL[0])
        print('save one non-parametric transpiled circ:',qasmF)
    

    # ----- submission ----------
    job =  backend.run(qcTL,shots=args.numShots, dynamic=True)
    jid=job.job_id()
    print('submitted JID=',jid,backend ,'\n do %d shots on %d qubits for %d circ, wait for execution ...'%(args.numShots,qcTL[0].num_qubits,len(qcTL)))

    if 0: # get circuit duration
        fix_me2
        print(type(job))
        circSL=job.circuits()
        print(circSL[2].qubit_duration(1))

    task.postprocessIBM_submit(job)
    jobMD=task.submitMeta    
           
    if args.verb>0: pprint(jobMD)
    

    if 1:  # debug when meta-data packing to JSON crahses
        import ruamel.yaml  as yaml
        ymlFd = open('dump.yaml', 'w')
        yaml.dump(jobMD, ymlFd, Dumper=yaml.CDumper)
        ymlFd.close() 


    if args.provider=='aer':  # shortcut for local simulation
        from toolbox.UQiskit_job import  harvest_retrievInfo, postprocess_job_results
        harvest_retrievInfo(job.result(),task.meta)
        rawCountsL=job.result().get_counts()
        if type(rawCountsL)!=type([]): rawCountsL=[rawCountsL] # Qiskit is inconsitent, requires this patch - I want always a list of results
        
        postprocess_job_results(rawCountsL,task.meta,task.expD)
        #...... WRITE  TASK  OUTPUT .........
        args.outPath=os.path.join(args.basePath,'meas')
        outF=os.path.join(args.outPath,jobMD['short_name']+'.h5')
        write4_data_hdf5(task.expD,outF,task.meta)
        anaCode= jobMD['analyzis']['ana_code']
        print('\n   ./%s --expName   %s    \n'%(anaCode,jobMD['short_name']))
        exit(0)

        
    #...... WRITE  JOB META-DATA .........
    outF=os.path.join(args.outPath,jobMD['short_name']+'.ibmq.h5')
    write4_data_hdf5(task.expD,outF,jobMD)
    print('M:end --expName   %s   %s  %s  jid=%s'%(jobMD['short_name'],jobMD['hash'],backend ,jobMD['submit']['job_id']))
    print('   ./retrieve_ibmq_job.py --expName   %s   \n'%(jobMD['short_name'] ))
   
