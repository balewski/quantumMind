#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Circuit Rewriting using the Transpiler
works w/ real backend

Updated 2022-05
based on :
https://lab.quantum-computing.ibm.com/user/5b7d94630feca300368b5419/lab/tree/qiskit-tutorials/qiskit/circuits_advanced/01_advanced_circuits.ipynb

https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/circuits_advanced/04_transpiler_passes_and_passmanager.ipynb


Do NOT use execute(), instead:
1) Circuits are rewritten to match the constraints of a given backend and optimized.
2) use backend to run circuit - if mapping is imposible compilation will fail

L0=Trivial layout: Map virtual qubits to the same numbered physical qubit on the device, i.e. [0,1,2,3,4] -> [0,1,2,3,4] (default in optimization_level=0).
L=1 Dense layout: Find the sub-graph of the device with same number of qubits as the circuit with the greatest connectivity (default in optimization_level=1).
L=2,3 Noise adaptive layout: Uses the noise properties of the device, in concert with the circuit properties, to generate the layout with the best noise properties (default in optimization_level=2 and optimization_level=3).


'''
# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint
import inspect  # python untility prints inspect.signature(class)

sys.path.append(os.path.abspath("../utils/"))
#from Circ_Plotter import Circ_Plotter
from Circ_Util import access_backend
import numpy as np
import qiskit as qk
from qiskit.visualization import plot_histogram, plot_gate_map, plot_circuit_layout
from qiskit.tools.monitor import job_monitor
from qiskit import IBMQ

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")

    parser.add_argument('-L','--optimLevel',type=int,default=3, help="transpiler ")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time and disable X-term")

    parser.add_argument('-b','--backName',default='ibmq_lima',help="backand for computations" )
    parser.add_argument('-q','--numQubit',type=int,default=3, help="num phys qubits")
    parser.add_argument('-n','--numShots',type=int,default=2002, help="shots")
    
    parser.add_argument( "-X","--no-Xterm", dest='noXterm', action='store_true', default=False, help="disable X-term for batch mode")


    args = parser.parse_args()
    args.prjName='ex5Transp'
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

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()


    print('\n\n ******  NEW circuit : GHZ  ****** ')
    circ_ghz=make_ghz_circ(args.numQubit)
    
    if args.verb>1:
        print('\nM:The transpile function has many input arguments:\n',inspect.signature(qk.transpile))

    '''Here we focus only on the most important ones: 
    - basis_gates, 
    - initial_layout 
    - optimization_level 
    - coupling_map    
    - seed_transpiler (int):
                sets random seed for the stochastic parts of the transpiler
    '''

    backend=access_backend(args.backName)
    print('base gates:', backend.configuration().basis_gates)
    #    assert not backend.configuration().simulator

    #plot_gate_map(backend, plot_directed=True, ax=plot.blank_page(10,'backend=%s'%backend))

    #https://qiskit.org/documentation/api/qiskit.visualization.plot_gate_map.html

    # https://qiskit.org/documentation/_modules/qiskit/compiler/transpile.html
    print(' Layout using optimization_level=',args.optimLevel)

    circT = qk.transpile(circ_ghz, backend=backend, optimization_level=args.optimLevel, seed_transpiler=args.rnd_seed, scheduling_method="alap")

    # qubit_duration only work with "scheduled circuit". So you first need to schedule the circuit. That can be done by specifying scheduling_method option in transpile. Please change your code like

    clock_dt=backend.configuration().dt
    print('duration of drive per qubit, clock dt/ns=%.2f'%(clock_dt*1e9))
    for qid in range(circT.num_qubits):
        len_dt=circT.qubit_duration(qid)
        len_us=len_dt*clock_dt*1e6
        print('q%d duration %.1f (us) =%d (dt)'%(qid,len_us,len_dt))


    print('circT Depth:', circT.depth())
    print('Gate counts:', circT.count_ops())

    print(circT)
    print('this was optimal circT from transpiler  for ',args.backName)

    # add Dynamical decoupling insertion pass.
    from qiskit.circuit.library import XGate
    from qiskit.transpiler import PassManager, InstructionDurations
    from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
    dd_sequence = [XGate(), XGate()]
    pm = PassManager([#ALAPSchedule(durations),
                  DynamicalDecoupling(durations, dd_sequence)])
    circ_dd = pm.run(circ)

    

    if not args.executeCircuit:
        print('NO execution of %d circuits, use -E to execute the job'%len(circT))
        exit(0)

    job =  backend.run(circT,shots=args.numShots)
    jid=job.job_id()

    print('submitted JID=',jid,backend ,' now wait for execution of your circuit ...')

    job_monitor(job)

    result = job.result()
    counts = result.get_counts(circT)
    print('counts:',args.backName); pprint(counts)
    resD={ k:counts.get(k) for k in counts.keys()}
    # alternative: counts.items() --> dict_items([('000', 196), ('001', 799), ('010', 1110), ...
    print('resD keys:',resD.keys())


    print("\n-----\n inspect some aspects of the submitted job:")
    print('\nbackend options:'); pprint(job.backend_options())
    print('\n job header'); pprint(job.header())
    circEL=job.circuits() # circuits as-executed
    circ4=circEL[0]
    print('numCirc=',len(circEL),circ4.name)
    print('exec circ depth:',circ4.depth(),', Gate counts:', circ4.count_ops())



