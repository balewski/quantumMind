#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
From Aziz:

Do not use 
 service = QiskitRuntimeService()
 backend = service.get_backend(backendN)

but use
from qiskit_ibm_provider import IBMProvider
provider = IBMProvider()
backend=provider.get_backend('ibm_hanoi')
job=backend.run(trans_circ, dynamic=True)
result=job.result()


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


'''

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options, Session
from quantum_serverless import get_arguments as serverless_arguments
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter
# convert the integers to bitstrings,if you want to apply the marginal_counts
from qiskit.result.utils import marginal_distribution
from qiskit.tools.visualization import circuit_drawer
   

from time import time, sleep
from qiskit.providers.jobstatus import JobStatus
import numpy as np
import h5py,os

import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num_shot',type=int,default=1000, help="shots")
    parser.add_argument('-b','--backend',default="ibmq_qasm_simulator", help="tasks")
    parser.add_argument('-a','--alpha', default=0.3, type=float, help='bias term , in [0,pi]')
    parser.add_argument('--spam_corr',type=int, default=1, help="Mitigate error associated with readout errors, 0=off")    
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))   
    return vars(args)

#...!...!....................
def circ_2q_nonlin():
    # Define the parameters alpha and theta
    alpha = Parameter('α')
    theta = Parameter('θ')
    # Create a Quantum Circuit with 2 qubits and 2 classical bits for measurement
    qc = QuantumCircuit(2, 2)
    qc.rx(alpha, 0)
    qc.ry(theta, 1)
    qc.cx(1, 0)
    qc.measure(0, 0)  
    # Add a conditional X gate on qubit 1 depending on the classical bit 0
    with qc.if_test((qc.cregs[0], 1)):
        qc.h(1)
    # Measure qubit 1 into classical bit 1
    qc.measure(1, 1)
    return qc, alpha, theta

#...!...!....................
def save_probs_h5(probD,qc,it):
    nb=qc.num_clbits
    MB=1<<nb
    measA=np.zeros(MB,dtype=np.float32)
    for key in probD:
        measA[int(key)]=probD[key]

    # Save this array to an HDF5 file
    coreN='patt1_t%d_%s.h5.tar'%(it,backendN)
    outF=os.path.join(outPath,coreN)
    print('saving locally:',coreN)
    with h5py.File(outF, 'w') as hdf:
        hdf.create_dataset('data_meas', data=measA)
        
    # Optionally, to verify the saved data
    with h5py.File(outF, 'r') as hdf:
        loaded_data = hdf['data_meas'][:]
        print(loaded_data)

#...!...!....................
def ana_exp(qcL,resultL):
    probL=[]   
    for ic in range(nCirc):
        probs = {}
        probD=job.result().quasi_dists[ic]
        for key, value in probD.items():
            probs[format(key,'0'+str(qc.num_clbits)+'b')] = value
        
        probq1=marginal_distribution(probs, indices=[1])
        if '1' not in probq1: probq1['1']=0.
        #print('aaa',probq1)
        probL.append(probq1['1'])
    # compare result w/ truth
    print('alpha=%.3f'%alpha_value)
    for ic in range(nCirc):
        theta=thetaV[ic]
        mprob=probL[ic]
        tprob=0.5- np.cos( alpha_value )/4. - np.cos(theta)/4.
        print('ic=%d  theta=%.3f   tprob=%.3f  mprob=%.3f'%(ic,theta,tprob,mprob))
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    sargs = serverless_arguments()
    if len(sargs)>0:
        backendN=sargs.get('backend')
        shots=sargs.get('num_shot')
        spamCorr=sargs.get('spam_corr')            
        outPath='/data'
    else:
        args=commandline_parser()
        backendN=args['backend']
        shots=args['num_shot']
        spamCorr=args['spam_corr']            
        outPath='out'

    assert spamCorr in [0,1]
    print('P:pattern1 start on',backendN)
    assert backendN in ["ibmq_qasm_simulator",
                        "ibmq_kolkata","ibmq_mumbai","ibm_algiers","ibm_hanoi", "ibm_cairo",# 27 qubits
                        "ibm_brisbane", # 127 qubits
                        ]

    service = QiskitRuntimeService()
 
    # Run on a simulator
    backend = service.get_backend(backendN)
 
    # https://qiskit.org/ecosystem/ibm-runtime/stubs/qiskit_ibm_runtime.options.Options.html#qiskit_ibm_runtime.options.Options
    
    options = Options()
    options.optimization_level=3  #  even heavier optimization (deafult=3)
    options.resilience_level = spamCorr  # Mitigate error associated with readout errors, (default=1, 0=off)
    options.execution.shots =shots
    session = Session(backend=backend)
    sampler = Sampler(session=session, options=options)

    T0=time()
    it=0
    print('\nP:pattern1 run task=%d ...'%it,backend)
    qcP, alpha, theta=circ_2q_nonlin()
    
    print(circuit_drawer(qcP, output='text',cregbundle=False))
    
    if 1: # Draw the circuit using the latex source
        latex_source = circuit_drawer(qcP, output='latex_source',cregbundle=False)
        print(latex_source)
        ok0
    
    alpha_value = np.pi*0.6  # Replace with the desired value for alpha
    alpha_value=args['alpha']
    nCirc=10
    thetaStep=np.pi/nCirc
    thetaV=np.arange(nCirc)*thetaStep
    qcL=[]
    for i in range(nCirc):        
        # Create a parameter binding dictionary
        param_dict = {alpha: alpha_value, theta: thetaV[i]}
        # Bind the values to the circuit
        qc = qcP.assign_parameters(param_dict)
        qcL.append(qc)
        
    print('do thetaV:',thetaV)
    # The bound_circuit is now ready to be executed
    
    print('P:run %d circuits ..., example:'%len(qcL));print(qc)
    job = sampler.run(qcL)
    elaT=time()-T0
    result = job.result()
    print('P:  ended elaT=%.1f sec\n probs:'%(elaT))
    for ic in range(nCirc):
        probD=result.quasi_dists[ic]
        print('ic=%d probD:%s'%(ic,probD))

    ana_exp(qcL,result)

    # https://qiskit.org/documentation/stubs/qiskit.visualization.circuit_drawer.html
    
    print('P:pattern1 done, shots:',shots,backend)
