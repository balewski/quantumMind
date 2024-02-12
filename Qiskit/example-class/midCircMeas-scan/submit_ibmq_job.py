#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Generates list of 1-q mid-circ-meas circuits to scan the whole chip
- no SPAM correction 
- no graphics
Submits job to IBMQ
Records meta-data containing  job_id 
HD5 arrays contain:
- initial and transpiled circ as array of strings, exported as qasm3
- input data(optional)


Dependence:  qiskit

Use cases:
??./submit_ibmq_job.py  -E  --provider aer -b aer_simulator
./submit_ibmq_job.py  -E  --provider aer -b fake_jakarta
./submit_ibmq_job.py  -E  --provider ibmq -b ibmq_qasm_simulator


'''

import sys,os
import time
from pprint import pprint

from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.UQiskit_job import access_qiskit_backend, submit_args_parser, remap_qubits, dump_qasm3

from TaskMidCircMeas1Q import TaskMidCircMeas1Q
from TaskTomoU3 import TaskTomoU3
import argparse

#...!...!..................
def get_parser(backName=None):  # add task-speciffic args
    parser = argparse.ArgumentParser()
    # TomoU3 or MidCircMeas task speciffic
    parser.add_argument('--initState', default=[2.4,0.6], type=float,  nargs='+', help='initial state defined by U3("theta/rad", "phi/rad", "lam=0") ')
    parser.add_argument('--qubitSelect',default='all',choices=['all','even','odd'],help="donw-select qubits to reduce number of circuit" )
    args=submit_args_parser(backName,parser)
    
    return args


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser(backName='aer_simulator')

    task=TaskMidCircMeas1Q(args)
    #1task=TaskTomoU3(args)
    
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

       
    if args.verb>1:    # decode  Qasm2 circuit
        for ic in range(2):        
            rec2=task.expD['transp_qasm3'][ic]#.decode("utf-8")            
            print(ic,'\nM:qasm3 %s:\n'%qcTL[ic].name,rec2)

        
    print('M: clone circuits over all %d qubits'%tot_qubits)
    qcTAL=qcTL.copy()
    for qid in range(1,tot_qubits):
        if args.backendName=='aer_simulator': break
        # tmp reduce num circ for Washington to avoid OOM
        if args.qubitSelect=='even' and qid%2==1: continue
        if args.qubitSelect=='odd' and qid%2==0: continue
        if qid>50: continue
        qMap=[ i for i in range(tot_qubits) ]
        qMap[0]=qid; qMap[qid]=0  # swap  measured qubit
        for qc in qcTL:
            qc1=remap_qubits(qc,qMap)
            qc1.name=qc1.name.replace('_q0','_q%d'%qid)
            #print('rem',qid,qc1.name)
            #print(qc1.draw(output="text", idle_wires=False))
            qcTAL.append(qc1)
            task.submitMeta['circ_sum']['circ_name'].append(qc1.name)
    task.submitMeta['circ_sum']['num_circ']=len(qcTAL)

    if 0:  #dump all transpiled qasm
       
        for j,qc in enumerate(qcTAL):
            print('# dump QASM3 %d circ:\n'%j)
            print(dump_qasm3(qc,backend))
            print('\n#  --- end ---\n')
        ok999
    
    # ----- submission ----------
    job =  backend.run(qcTAL,shots=args.numShots, dynamic=True) 
    jid=job.job_id()
    print('submitted JID=',jid,backend ,'\n do %d shots on %d qubits for %d, wait for execution ...'%(args.numShots,qcTAL[-1].num_qubits,len(qcTAL)))

   
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
   
