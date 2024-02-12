#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Uses:
  sampler = Sampler(session=session, options=options)
  job = sampler.run(circuit)
  probs= job.result().quasi_dists
INPUT:
  nTask=args.get('num_task')
  shots=args.get('num_shot')
OUTPUT:
  hd5 output file to be transported back to my laptop

Issues:
 - executed task is defined by job = serverless.run(title) instead of the 'entrypoint'.
if I change title in  QiskitPattern(..) but not match it in serverless.run() my old .py will be executed - It took me 20 min to figure out why I see the old results.
- files created in /data on your server stay beyond lifetime of the job. If I run new job  I still see  & can download the old files and the new files - it is confusing.
- why can't I downaload .h5 files created by the task? It is a pain to  write .tar files from python. Now I just name my HDF5 files: abc.h5.tar to fool the interface.


Instruction
https://qiskit-extensions.github.io/quantum-serverless/migration/migration_from_qiskit_runtime_programs.html

 https://qiskit.org/ecosystem/ibm-runtime/stubs/qiskit_ibm_runtime.options.Options.html#qiskit_ibm_runtime.options.Options

'''

#from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options, Session
import quantum_serverless as servless

from time import time, sleep
from qiskit.providers.jobstatus import JobStatus
import numpy as np
import h5py,os


import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num_shot',type=int,default=600, help="shots")
    parser.add_argument('-t','--num_task',type=int,default=2, help="tasks")
    parser.add_argument('-b','--backend',default="ibmq_qasm_simulator", help="tasks")    
    parser.add_argument('--spam_corr',type=int, default=1, help="Mitigate error associated with readout errors, 0=off")
    parser.add_argument("--out_path",default='out/',help="all outputs from  experiment")
    args = parser.parse_args()
    import hashlib
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    args.short_name='ghz_'+myHN
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert args.spam_corr in [0,1]

    return vars(args)

#...!...!....................
def create_ghz_circuit(n):
    circuit = QuantumCircuit(n)
    # Apply Hadamard gate to the first qubit
    circuit.h(0)
    # Apply CNOT gates
    for i in range(1, n):
        circuit.cx(0, i)
    circuit.measure_all()
    return circuit

#...!...!....................
def save_probs_h5(probD,qc,it):
    nb=qc.num_clbits
    MB=1<<nb
    measA=np.zeros(MB,dtype=np.float32)
    for key in probD:
        measA[int(key)]=probD[key]
    # Create a random numpy array
    dataRnd = np.random.rand(5, 3)  # Example array

    # Save this array to an HDF5 file
    coreN='%s_%d.h5'%(args['short_name'],it)
    outF=os.path.join(args['out_path'],coreN)
    print('saving locally:',coreN)
    with h5py.File(outF, 'w') as hdf:
        hdf.create_dataset('data_meas', data=measA)
        hdf.create_dataset('data_random', data=dataRnd)

    tarL.append(coreN)
    # Optionally, to verify the saved data
    with h5py.File(outF, 'r') as hdf:
        loaded_data = hdf['data_meas'][:]
        print(loaded_data)  

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":

    args = servless.get_arguments()    
    
    if len(args)==0:
        args=commandline_parser()
    else:  # exceptions:
        args['out_path']='/data'
         
    print('P:pattern1 short_name=%s start on %s'%(args['short_name'],args['backend']))
 
    service = QiskitRuntimeService()
 
    # Run on a simulator
    backend = service.get_backend(args['backend'])   
    
    options = Options()
    options.optimization_level=3  #  even heavier optimization (deafult=3)
    options.resilience_level = args['spam_corr']  # Mitigate error associated with readout errors, (default=1, 0=off)
    options.execution.shots =args['num_shot']
    session = Session(backend=backend)
    sampler = Sampler(session=session, options=options)

    nTask=args['num_task']
    T0=time()
    T1=T0
    tarL=[]
    for it in range(nTask):
        print('\nP:pattern1 run task=%d ...'%it,backend)
        circuit=create_ghz_circuit(it+2)
        print('P:new circ, it:',it);print(circuit)
        job = sampler.run(circuit)
        i=0
        while True:
            jstat=job.status()
            elaT=time()-T0
            print('P:i=%d task=%d status=%s, elaT=%.1f sec'%(i,it,jstat,elaT))
            if jstat==JobStatus.DONE: break
            i+=1; sleep(10)
        jstat=job.status()
        T2=time()
        elaT=T2-T0
        print('P: task=%d end-status=%s, elaT=%.1f sec,  taskDelay=%.1f sec'%(it,jstat,elaT,T2-T1))
        T1=T2
        if jstat==JobStatus.DONE :
            result = job.result()
            elaT=time()-T0
            probD=result.quasi_dists[0] # just 1 circuit
            print('P: task=%d ended elaT=%.1f sec\n probs:'%(it,elaT), probD)
            save_probs_h5(probD,circuit,it)
        else:
            print('P:job failed',jstat)
    MD={}
    MD["tar_list"]= ','.join(tarL)
    servless.save_result(MD)
    print('P:pattern1 done')
