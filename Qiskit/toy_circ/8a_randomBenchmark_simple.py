#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
execute RB  using noisy simulator
no read-err-corr

Based on
https://qiskit.org/documentation/experiments/tutorials/randomized_benchmarking.html
'''

import time,os,sys
from pprint import pprint
import numpy as np

from qiskit_experiments.library import StandardRB, InterleavedRB
from qiskit_experiments.framework import ParallelExperiment, BatchExperiment
import qiskit.circuit.library as circuits

# For simulation
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeLima

# for plotting
sys.path.append(os.path.abspath("../"))
from utils.PlotterBackbone import PlotterBackbone

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',action='store_true', default=False,help="disable X-term for batch mode")   
    parser.add_argument('-o',"--outPath",default='out',help="output path ")
    
    parser.add_argument('-n','--numShots', default=10, type=int, help='num of shots')
    parser.add_argument('-q','--qubits', default=[1,3], type=int,  nargs='+', help='1 or 2 qubits, space separated ') 
    
    args = parser.parse_args()
            
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    assert len(args.qubits) in [1,2]  # other lengths are undefined
    return args



#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":

    args=get_parser()

    backend = AerSimulator.from_backend(FakeLima())
    # Clifford seqences
    lengths= [1, 2, 5, 10, 20, 50, 75, 100,  150,  200 ,400]    
    
    # Run an RB experiment on qubit 0
    exp1 = StandardRB(args.qubits, lengths, num_samples=args.numShots, seed=111)
    expdata1 = exp1.run(backend).block_for_results()
    results1 = expdata1.analysis_results()
    print(type(expdata1))  # qiskit_experiments.framework.experiment_data.ExperimentData
    
    # View result data
    print("Gate error ratio: %s" % expdata1.experiment.analysis.options.gate_error_ratio)
    print('M: num results',len(results1))
    for result in results1:
        print(result)

    # ----  just plotting
    args.prjName='myRB_q'+'-'.join( str(i) for i in args.qubits)
    plot=PlotterBackbone(args)
    rbFitFig = expdata1.figure(0).figure
    plot.canvas4figure( rbFitFig, figId=5 )  # from Aziz, IBM Quantum Support
    
    # modify the plot so it looks nicer - if you want
    ax= rbFitFig.axes[0]
    if len(args.qubits) ==2:
        ax.set_xlim(0.9,) ;    ax.set_xscale('log')
    tit='%s,   qubits:%s,  shots=%d'%(backend,str(args.qubits),args.numShots)
    ax.set(title=tit)
    plot.display_all()  # now all plots are created
   
    print('\nEND-OK')

