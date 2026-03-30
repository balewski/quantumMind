#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import os, secrets
from time import time, localtime
from qiskit import QuantumCircuit

import argparse  
from pprint import pprint
from toolbox.Util_NumpyIOv1 import write_data_npz, read_data_npz
from toolbox.Util_IOfunc import dateT2Str

# Explicit Guppy imports
from guppylang import guppy
from guppylang.std.builtins import array, comptime, result
from guppylang.std.quantum import qubit, h, cx, rz, crz, measure_array, pi

from toolbox.Util_Guppy import guppy_to_qiskit

import qnexus as qnx

# Suppress qiskit transpiler logs
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qnexus.models").setLevel(logging.ERROR)

def commandline_parser(backName="ideal", provName="Qtuum_cloud"):
    parser = argparse.ArgumentParser()

    parser.add_argument('-q','--num_qubits', type=int, default=3, help='Size of circut - very limitted')
    parser.add_argument('-n','--num_shots', type=int, default=100, help='Number of shots')
    parser.add_argument('-b','--backend', type=str, choices=['ideal', 'noisy','Helios-1E'], default=backName, help='Simulation backend')
    

    parser.add_argument('-v', '--verb', type=int, default=1, help='Verbosity level')
    parser.add_argument('-t','--taskType', nargs='+', type=str, default=['ghz'], help='task type and optional params, e.g. "qft 3 "')
    parser.add_argument('-c',"--printCirc", type=str, default="", help="i=ideal, d=decomposed, t=transpiled")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")
    parser.add_argument("--basePath",default='out',help="head dir for set of experiments")
    parser.add_argument("--expName",  default=None,help='(optional) replaces job ID assigned during submission by users choice')

    #add_noisy_backend_args(parser)
    parser.add_argument('--noise_nuclDepol', type=float, default=1e-2, help=' depolarizing error')

    args = parser.parse_args()
    if args.backend is None: 
        args.backend = backName
    args.provider = provName

    args.outPath=os.path.join(args.basePath,'jobs')
    assert os.path.exists(args.outPath), f"missing outPath: {args.outPath}"
    
    # Ensure taskType is always a list for downstream code
    if isinstance(args.taskType, str):
        args.taskType = [args.taskType]    
        
    
    for arg in vars(args):
        if 'noisy' not in args.backend and 'noise' in arg: 
            continue
        print('myArgs:', arg, getattr(args, arg))

    return args


#...!...!....................
def upload_qtuum_circuit(gu_prog,md):
   
    pmd=md['payload']
    sbm=md['submit']
    sbm['user_group']='CHM170'
    sbm['project_name']='mar_guppy2'
    sbm['backend']=args.backend
    
    #.... add more meta-date
    if args.expName==None:
        tag='emu' if args.backend=='Helios-1E' else 'hw'
        md['short_name']='%s_%s'%(tag,md['hash'])
    else:
        md['short_name']=args.expName
    sbm['hugr_tag']=secrets.token_hex(8)
    pprint(md)
    
    project = qnx.projects.get_or_create(name=sbm['project_name'])
    qnx.context.set_active_project(project)
    devConf = qnx.QuantinuumConfig(device_name=sbm['backend'], user_group=sbm['user_group'],max_cost=MAX_COST_HQC, compiler_options={"max-qubits": pmd['num_qubits']})

    #1print('use devConf:',devConf)

    # Compile to HUGR IR
    hugr_pkg = gu_prog.compile()  # compile the full program
    print('hugr done')

    # 3) Upload HUGR to Nexus
      
    t0 = time()
    refU = qnx.hugr.upload(
        hugr_package=hugr_pkg,
        name=sbm['hugr_tag'],
        description="abc123",
        project=project,
    )  # returns a HUGR reference usable in execute jobs 
    t1 = time()
    print('elaT=%.1f sec, HUGR uploaded'%(t1-t0))
    return refU,devConf

#...!...!....................
def submit_qtuum_programs(refUL,devConf,md):
    nCirc=len(refUL)
    sbm=md['submit']
    sbm['exec_tag']="exec_"+sbm['hugr_tag']
    t0=time()
    shotL=[sbm['num_shots']]*nCirc
    ref_exec= qnx.start_execute_job( programs=refUL, n_shots=shotL,
                                     backend_config=devConf,name=sbm['exec_tag'])
    t1=time()
    print('nCirc=%d  submit elaT=%.1f  hash=%s\n'%(nCirc,t1-t0,sbm['exec_tag']))

    sbm['job_ref_json']=ref_exec.model_dump_json()
    
    #.... harvest meta data
    t1=localtime()
    sbm['date']=dateT2Str(t1)
    sbm['unix_time']=int(time())
    sbm['num_circ']=nCirc
    pprint(sbm)
    
#...!...!....................
def buildMetaZoo(args):
    pmd={}  # payload
    # args.taskType is a list: first element is type, rest are params
    pmd['task_type'] = args.taskType[0]
    pmd['task_param'] = args.taskType[1:]
    pmd['num_qubits']=args.num_qubits
        
    sbm={}
    sbm['backend']=args.backend
    sbm['provider']=args.provider
    sbm['num_shots']=args.num_shots
    if args.backend=='noisy':
        sbm['lsf_nuc_depol']=args.noise_nuclDepol
        sbm['lsf_xdephase']=args.noise_xDephase

    pom={}
        
    md={ 'payload':pmd, 'submit':sbm , 'postproc':pom}

    myHN=secrets.token_hex(3)
    md['hash']=myHN

    if args.expName==None:
        md['short_name']='%s_%s'%(args.backend,md['hash'])
    else:
        md['short_name']=args.expName
    return md
                                                              
def generate_ghz_program(num_qubits: int):
    """
    Factory function to build a Guppy program that prepares an N-qubit GHZ state.
    """
    @guppy
    def main_ghz() -> None:
        # Allocate array of N qubits using comptime
        qs = array(qubit() for _ in range(comptime(num_qubits)))
        h(qs[0])
        for i in range(comptime(num_qubits - 1)):
            cx(qs[i], qs[i + 1])
        result("c", measure_array(qs))
    return main_ghz

def generate_qft_bench(num_qubits: int, task_params: list):
    """
    Factory function to build a Guppy QFT benchmark program. 
    task_params: list where the first element is the input integer k.
    """
    assert len(task_params)==1
    inpInt = int(task_params[0])
    
    @guppy
    def iqft_n(qs: array[qubit, comptime(num_qubits)]) -> None:
        # 1. Reverse qubit order with physical SWAP gates
        for i in range(comptime(num_qubits // 2)):
            cx(qs[i], qs[comptime(num_qubits) - 1 - i])
            cx(qs[comptime(num_qubits) - 1 - i], qs[i])
            cx(qs[i], qs[comptime(num_qubits) - 1 - i])
        
        # 2. Iterate through qubits and apply rotations and H gates
        for i in range(comptime(num_qubits)):
            for j in range(i):
                phi = -pi / (2 ** (i - j))
                rz(qs[j], phi / 2)
                crz(qs[j], qs[i], phi)
            h(qs[i])

    @guppy
    def qft_prep_n(qs: array[qubit, comptime(num_qubits)], inpInt: int) -> None:
        """Prepares Fourier state corresponding to computational state |inpInt|."""
        for i in range(comptime(num_qubits)):
            h(qs[i])
        for j in range(comptime(num_qubits)):
            rz(qs[j], (2.0 * pi * inpInt) / (2 ** (comptime(num_qubits) - j)))

    @guppy
    def main_qft_bench() -> None:
        val = comptime(inpInt)
        qs = array(qubit() for _ in range(comptime(num_qubits)))
        qft_prep_n(qs, val)
        iqft_n(qs)
        result("c", measure_array(qs))
        
    return main_qft_bench



if __name__ == '__main__':
    MAX_COST_HQC=2000
    np.set_printoptions(precision=3)
    
    args = commandline_parser(backName='ideal')
    expMD=buildMetaZoo(args)

    # args.taskType is now a list; first element selects the circuit type
    tt = args.taskType[0] if isinstance(args.taskType, list) else args.taskType
    if tt == 'ghz':
        gu_prog = generate_ghz_program(args.num_qubits)
        expD={}
    elif tt == 'qft':
        gu_prog = generate_qft_bench(args.num_qubits, args.taskType[1:])
        expD={}
    else:
        raise ValueError(f"Unknown taskType: {tt}")

    # Static type check and compile
    gu_prog.check()
 

    # print the circuits using -->TKet -->Qiskit transformations
    circQi=guppy_to_qiskit(gu_prog,nq=args.num_qubits)

    if 'i' in args.printCirc:
        print(circQi.draw())
    print('ideal gates:',circQi.count_ops())    
 
    
    if args.backend=='ideal':
        print('--- 2. Setting up Selene Emulator ...')
        t0 = time()
        # We must inform the emulator of the total qubit allocation needed
        emuBase = gu_prog.emulator(n_qubits=args.num_qubits)
   
        print("Selected: Ideal Statevector Simulator")
        runner = emuBase.statevector_sim()
        t1 = time()
        print(f'elaT={t1-t0:.1f} sec...')
        
        print('--- 3. Executing Circuit ...')
        t0 = time()
    
        simResults = runner.run()
        
        t1 = time()
        shots_list = simResults.collated_shots()
     
        if len(shots_list) > 0:
            print(f"Debug - Raw first shot looks like: {shots_list[0]}")
        print('M: done device:',args.backend,'\n')
        os._exit(0)

    # ----- submission ----------
    numShots=expMD['submit']['num_shots']
    print('M:job starting,  nq=%d  shots=%d at %s  ...'%(args.num_qubits,numShots,args.backend))
    refU,devConf=upload_qtuum_circuit(gu_prog,expMD)
    submit_qtuum_programs([refU],devConf,expMD)

    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,expMD['short_name']+'.qtuum.npz')
    write_data_npz(expD,outF,expMD)
    print('M:end --expName   %s   %s  %s '%(expMD['short_name'],expMD['hash'], args.backend))
    print('   ./retrieve_qtuum_job.py --expName   %s   \n'%(expMD['short_name'] ))
    os._exit(0)

    
