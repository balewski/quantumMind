#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Run on  custom noisy simulator

No graphics

Updated 2022-09
based on :


'''

import time,os,sys
from pprint import pprint

import numpy as np
import qiskit as qk

from qiskit.tools.monitor import job_monitor
from qiskit import IBMQ

# For simulation
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeLima

#-------------------  define noisy simulator model --------------
# noisy simulator, build based on:
# https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, thermal_relaxation_error

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time and disables X-term")
    parser.add_argument('-q','--numQubit',type=int,default=3, help="num ideal qubits")
    parser.add_argument('-n','--numShots',type=int,default=2002, help="shots")

    args = parser.parse_args()

    args.rnd_seed=111 # for transpiler
    if args.executeCircuit: args.noXterm=True    

    print('qiskit ver=',qk.__qiskit_version__)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def make_ghz_circ(nq):
    name='ghz_%dq'%nq
    ghz = qk.QuantumCircuit(nq, nq,name=name)
    ghz.h(0)
    for idx in range(1,nq):
        ghz.cx(0,idx)
    ghz.barrier(range(nq))
    ghz.measure(range(nq), range(nq))
    print(ghz)
    return ghz

#............................
def config_noise_model_ibmq_all2all(md):
    # paramaters must match NoiseModel() from UQiskit_noisySim.py
    # setting of IBMQ nosie matched to avearge for guadalupe, on 2022-09-01
    # https://quantum-computing.ibm.com/services/resources?tab=systems&system=ibmq_guadalupe
    
    cbf={}  # bit-flip noise config
    cbf['prob_meas']=0.025
    cbf['prob_u3']  =0.0004
    cbf['prob_cx']  =0.014

    ctr={} # thermal noise config
    ctr['time_u3'] = 60 # (two X90 pulses)
    ctr['time_cx'] = 450  # (nSec)
    ctr['T1/T2']=[ 100e3, 100e3 ] # (nSec) , remember : T1 >= T2

    if 0: # U2-gates are not used
        cbf['prob_u2']= 0.01
        ctr['time_u2'] = 50  # (single X90 pulse)
    
    cf={ 'bit_flip':cbf, 'thermal':ctr, 'name':'ibmq'} # noisy model configuration
    md['noise_model']=cf

#...!...!....................
def access_noisySim_backend(md, verb=1):  # md must contain noisy simulator parameterization
    pprint(md)
    noise_model = NoiseModel()
    cf=md['noise_model']
    print('ww',cf)
    if 1:  # enable all noise types
        my_measure_noise(noise_model,cf['bit_flip'])
        my_u3_noise(noise_model,cf)
        my_cx_noise(noise_model,cf)
    backend =AerSimulator(noise_model=noise_model)
    return backend


#............................
def my_measure_noise(noise_m,cf):  # measurement noise model
    p_meas = cf['prob_meas']
    # QuantumError objects
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    # Add errors to noise model
    noise_m.add_all_qubit_quantum_error(error_meas, "measure")


#............................
def my_u3_noise(noise_m,cf):  # U3 : bit-flip & T1/T2 noise model 
     p_u3 =cf['bit_flip']['prob_u3']
     error_flip = pauli_error([('X',p_u3), ('I', 1 - p_u3)])

     T1,T2  =cf['thermal']['T1/T2']
     time_u3=cf['thermal']['time_u3']
     error_all  = error_flip.compose(thermal_relaxation_error(T1, T2, time_u3))
     noise_m.add_all_qubit_quantum_error(error_all, ["u3"])
     
#............................
def my_cx_noise(noise_m,cf):  # CX : bit-flip & T1/T2 noise model 
     p_cx =cf['bit_flip']['prob_cx']
     error_half = pauli_error([('X',p_cx), ('I', 1 - p_cx)])
     error_flip = error_half.tensor(error_half)

     T1,T2  =cf['thermal']['T1/T2']
     time_cx=cf['thermal']['time_u3']
     error_all  = error_flip.compose(thermal_relaxation_error(T1, T2, time_cx))
     noise_m.add_all_qubit_quantum_error(error_all, ["cx"])
     
    

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    print('\n\n ******  NEW circuit : GHZ  ****** ')
    circ=make_ghz_circ(args.numQubit)

    jobMD={}

    #------ setup noisy backend -----
    config_noise_model_ibmq_all2all(jobMD)
  
    backend=access_noisySim_backend(jobMD,args.verb)

    circT = qk.transpile(circ, backend=backend)
    
    print('circT Depth:', circT.depth())
    print('Gate counts:', circT.count_ops())
    print('this was optimal circT from transpiler  for ',backend)

    if not args.executeCircuit:
        print('NO execution of circuit, use -E to execute the job')
        exit(0)

    
    # - - - -  FIRE JOB - - - - - - -
    job =  backend.run(circT,shots=args.numShots)
    jid=job.job_id()
    print('submitted JID=',jid,backend ,' now wait for execution of your circuit ...')
    job_monitor(job)
    result = job.result()
            
    counts = result.get_counts(circT)
    print('counts:',backend); pprint(counts)
    resD={ k:counts.get(k) for k in counts.keys()}
    # alternative: counts.items() --> dict_items([('000', 196), ('001', 799), ('010', 1110), ...
    print('resD keys:',resD.keys())

    print('M:done')
