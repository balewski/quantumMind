#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Submit 1 circuit to Quantinuum Nexus - CAREFULL w/ credits
Updated to use the newer Nexus interface (qnexus)
'''

import os, secrets
import qnexus as qnx
from pytket import Circuit
from pytket.circuit import BasisOrder
from pytket.extensions.qiskit import tk_to_qiskit # only for printing
from pprint import pprint
from time import time, sleep

#...!...!..................
def create_circ(N=1):  # TKET circ
    qc = Circuit(1, 1)
    for i in range(N):
        qc.X(0)
    qc.H(0)    
    qc.measure_all()
    return qc

#=================================
#  M A I N 
#=================================
if __name__ == "__main__":

    # qnx.login_with_credentials() # call once to authenticate
    
    myTag = '_' + secrets.token_hex(3)
    shots = 10
    #devName = "H1-2LE"  # noiseless simulation of H2.
    devName = "H2-1E" # Use for error-modelled emulation of H2.
    
    #myAccount = 'CSC641'
    myAccount = 'CHM170'
    project = qnx.projects.get_or_create(name="qcrank-feb-14b")
    qnx.context.set_active_project(project)

    print('define some TKET circ')
    qc = create_circ(3)
    print(tk_to_qiskit(qc))
    print("\n--- Gate Sequence in TKet---")
    print(qc)
    for command in qc.get_commands():
        print(command)
    
    print('\nuploading circuit ...')
    t0 = time()
    ref = qnx.circuits.upload(circuit=qc, name='myca'+myTag)
    t1 = time()
    print('elaT=%.1f sec, uploaded, compiling ...'%(t1-t0))

    devConf = qnx.QuantinuumConfig(device_name=devName, user_group=myAccount) 
    print('use devConf:', devConf)
    
    t0 = time()
    #...  compile 1 circ
    # qnx.compile returns a collection of CircuitRef
    refC_list = qnx.compile(programs=[ref], name='comp'+myTag,
                            optimisation_level=2, backend_config=devConf,
                            project=project)
    refC = refC_list[0]
    t1 = time()
    print('elaT=%.1f sec, compiled, executing ...'%(t1-t0))

    # Optional: download and print the compiled circuit commands
    qcT = refC.download_circuit()
    print('\ncommands for the compiled circuit on %s:'%devName)
    for x in qcT.get_commands():
        print(x)

    #... get cost
    cost = qnx.circuits.cost(circuit_ref=refC, n_shots=shots,
                             backend_config=devConf, syntax_checker="H1-1SC")
    print('\nshots=%d cost=%.1f:'%(shots, cost))

    #.... execution     
    t0 = time()
    ref_exec = qnx.start_execute_job(programs=[refC], n_shots=[shots],
                                     backend_config=devConf, name="exec"+myTag)    
    t1 = time()
    print('job submit elaT=%.1f, waiting for results ...'%(t1-t0))
    
    qnx.jobs.wait_for(ref_exec)
    results = qnx.jobs.results(ref_exec)
    t2 = time()
    print('execution finished, total elaT=%.1f\n'%(t2-t1))
    
    result = results[0].download_result()
    
    # use this to match Qiskit bit order for QCrank
    tket_basis = BasisOrder.dlo # DLO gives (c[1], c[0]) == (1, 0)
    tket_counts = result.get_counts(basis=tket_basis)
    
    print('\nQCrank Tket counts:'); pprint(tket_counts)
    print('\ndone devName=', devName)
    print('status:', qnx.jobs.status(ref_exec))
