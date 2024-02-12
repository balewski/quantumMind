#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Run on FakSimulator or real HW
Uses pattern:
  circT = qk.transpile(circ_ghz, backend=backend, optimization_level=args.optimLevel)
  job =  backend.run(circT,shots=args.numShots)
  job_monitor(job)
  result = job.result()
No graphics

Updated 2022-05
based on :
https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/simulators/2_device_noise_simulation.ipynb

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


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-L','--optimLevel',type=int,default=3, help="transpiler ")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time ")
    parser.add_argument('-q','--numQubit',type=int,default=3, help="num ideal qubits")
    parser.add_argument('-n','--numShots',type=int,default=2002, help="shots")

    args = parser.parse_args()

    args.rnd_seed=111 # for transpiler

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

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    print('\n\n ******  NEW circuit : GHZ  ****** ')
    circ_ghz=make_ghz_circ(args.numQubit)
    backend = AerSimulator.from_backend(FakeLima())
    
    print('base gates:', backend.configuration().basis_gates)

    print(' Layout using optimization_level=',args.optimLevel)
    circT = qk.transpile(circ_ghz, backend=backend, optimization_level=args.optimLevel, seed_transpiler=args.rnd_seed)

    tit='optimLevel=%d, %s'%(args.optimLevel,backend)
    print('circT Depth:', circT.depth(), tit)
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
