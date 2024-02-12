#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Running a QiskitPattern on IBM server

https://qiskit-extensions.github.io/quantum-serverless/getting_started/basic/01_running_program.html

Source code

https://qiskit-extensions.github.io/quantum-serverless/stubs/quantum_serverless.core.IBMServerlessProvider.html#quantum_serverless.core.IBMServerlessProvider.__init__

https://qiskit-extensions.github.io/quantum-serverless/stubs/quantum_serverless.core.QiskitPattern.html

Instruction
https://qiskit-extensions.github.io/quantum-serverless/migration/migration_from_qiskit_runtime_programs.html

'''

import os
from time import time, sleep
from quantum_serverless import IBMServerlessProvider
from quantum_serverless import QiskitPattern
import hashlib

import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-n','--numShot',type=int,default=300, help="shots per circuit")

    parser.add_argument('-b','--backend',default="ibmq_qasm_simulator", help="QPU running job")
    parser.add_argument("--expName",  default=None,help='(optional)replaces IBMQ jobID assigned during submission by users choice')
    parser.add_argument('--spam_corr',type=int, default=1, help="Mitigate error associated with readout errors, 0=off")
   
    
    args = parser.parse_args()
    args.outPath='out'
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert args.spam_corr in [0,1]
    if 1:
        assert args.backend in [
            "ibmq_qasm_simulator",
            "ibmq_kolkata","ibmq_mumbai","ibm_algiers","ibm_hanoi", "ibm_cairo",  # 27 qubits
            "ibm_brisbane", "ibm_cusco","ibm_osaka" # 127 qubits
                        ]

    return args


    
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=commandline_parser()

    myArgs={'num_shot':args.numShot,'backend':args.backend}
    myArgs['num_task']=3
    myArgs['spam_corr']=args.spam_corr
    if args.expName!=None:
        myArgs['short_name']=args.expName
    else: 
        myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
        myArgs['short_name']='hybr_%s_%s'%(myHN,args.backend.split('_')[-1])
    
    # Authenticate to the IBM serverless provider
    serverless = IBMServerlessProvider() # run once:  /dataVault/activate_IBMProvider.py

    print('S:serverless acquired',serverless)
    
    pattTit=myArgs['short_name']  # this is primary job identiffier (not the .py file)
    myCode="pattern1_sampler.py"
    myCode="pattern3_backRun.py"
    #myCode='run_2q_nonlin.py'
    myPath='./src/'

    print('S: tit=%s  entry=%s  path=%s'%(pattTit,myCode, myPath))

    # Deploy the workflow
    pattern = QiskitPattern(
        title=pattTit,
        entrypoint=myCode,
        working_dir=myPath,
        dependencies=["scikit-learn==1.2.1","h5py==3.10.0"]
        )
    # you need to specify  the dependencies you want to install like scikit-learn

    serverless.upload(pattern )


    print('S: pattern %s  uploaded, submit serverless, myArgs:'%pattTit,myArgs)

    # Run my workflow remotely
    T0=time()
    job = serverless.run(pattTit,arguments=myArgs)
    elaT=time()-T0
    print('S: job submitted elaT=%.1f sec'%elaT)

    # Retrieve status, logs, results
    for k in range(1000):
        jstat=job.status()
        elaT=time()-T0
        print('S:k=%d status=%s, elaT=%.1f sec'%(k,jstat,elaT))
        if jstat in ['QUEUED','PENDING','RUNNING']:
            sleep(20) ;     continue
        break
    sleep(5) #?? Message: Http bad request., Code: 403

    jobLogD=job.logs()
    print('\nS:logs:',jobLogD)
    jobResD=job.result()
    print('\nS:results:',jobResD)
    #>>>S:results: {'tar_list': 'None_0.h5.tar,None_1.h5.tar,None_2.h5.tar'}
    # import produced data, only .tar

    #1 out_files = serverless.files()  # too much junk is pulled
    out_files=jobResD['tar_list'].split(',')
    print('S:out_files;',out_files)
    if len(out_files) > 0:
        for x in out_files:
            print('S:download %s'%(x))
            serverless.file_download(x, download_location='out')

    elaT=time()-T0
    print('S: done elaT=%.1f sec'%elaT)

