#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 Retrieve  results of IBM job
 Use case:
 ./retrieve_qart_job.py --expName ideal_3fbad8
'''

import os, sys
import numpy as np
from pprint import pprint
from time import time, sleep
import json
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qnexus.models").setLevel(logging.ERROR)
import qnexus as qnx

from qnexus.models.references import ExecuteJobRef

from toolbox.Util_NumpyIOv1 import write_data_npz, read_data_npz
#from toolbox.Util_QArt import  get_QArt_backend, applySPAMv2
#from qiskit.providers import JobStatus

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase output verbosity", default=1)
    parser.add_argument("--basePath",default='out',help="head dir for set of experiments")
    parser.add_argument('-e',"--expName",  default=None, help='Experiment name assigned during submission')
    
    parser.add_argument( "--cancelJob", action='store_true', default=False, help="use with caution ")

    args = parser.parse_args()
    args.inpPath=os.path.join(args.basePath,'jobs')
    args.outPath=os.path.join(args.basePath,'meas')
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def retrieve_qtuum_job(md,bigD):
    #pprint(md)
    sbm=md['submit']
    pmd=md['payload']
    #print(sorted(md))

    data = json.loads(sbm['job_ref_json'])
    ref_exec= ExecuteJobRef(**data)

    jobStatus=qnx.jobs.status(ref_exec)
    if args.verb>1:
        print('\nstatus3:',jobStatus)

    stat=jobStatus.status
    print('stat:',stat) #,jobStatus.running_time)

    qnx.jobs.wait_for(ref_exec)
    results = qnx.jobs.results(ref_exec)

    nCirc=len(results)
    print('job  finished, nCirc=%d'%(nCirc))


    qa = {
        'num_shots': sbm.get('num_shots', 0),
        'cost_hqc': getattr(jobStatus, 'cost', 0.0),
        'job_id': ref_exec.id,
    }
    qa['status']=str(stat)
    qa['num_circ']=nCirc
   
    #qa['timestamp_running']=execTimeConverter(data)

    countsL=[None]*nCirc
    for ic in range(nCirc):
        result_nx = results[ic].download_result()
        counts_nx = result_nx.collated_counts()
        # Extract bitstrings. Keys are tuples of (tag, bitstring) pairs.
        bitstring_dict = {"".join(tag_res[1] for tag_res in key)[::-1]: val for key, val in counts_nx.items()}
        countsL[ic]=bitstring_dict
        #print('\nis=%d  res:'%(ic)); pprint(bitstring_dict)

    print('job QA'); pprint(qa)
    md['job_qa']=qa
    
    # Define variables for postprocessing
    n_qubits = pmd['num_qubits']
    n_shots = sbm['num_shots']
    counts = countsL[0]  # Use the first circuit results

    def postproc_ghz(counts_dict, n_qubits_loc, n_shots_loc):
        state0 = '0' * n_qubits_loc
        state1 = '1' * n_qubits_loc
        prob_loc = (counts_dict.get(state0, 0) + counts_dict.get(state1, 0)) / n_shots_loc
        perr_loc = np.sqrt(prob_loc * (1 - prob_loc) / n_shots_loc)
        print(f"GHZ nq:{n_qubits_loc} Fidelity : {prob_loc:.4f} +/- {perr_loc:.4f}  fail: {1-prob_loc:.4f}  job:{md['short_name']}\n")
        return np.array([prob_loc, perr_loc])

    def postproc_qft(counts_dict, n_qubits_loc, n_shots_loc, pmd):
        # task_param: list-like, first element is integer frequency k
        inpInt = int(pmd['task_param'][0])
        target_str = f"{inpInt:0{n_qubits_loc}b}"
        success_count = counts_dict.get(target_str, 0)
        prob_loc = success_count / n_shots_loc
        perr_loc = np.sqrt(prob_loc * (1 - prob_loc) / n_shots_loc)
        # compute incorrect outcomes summary
        wrong_total_counts = n_shots_loc -  success_count
        any_states=len(counts_dict)
        wrong_states = any_states-1
        wrong_prob_loc = wrong_total_counts / n_shots_loc
        print(f"Incorrect states: {wrong_states} of {any_states}, total probability: {wrong_prob_loc:.4f} ({wrong_total_counts:.1f}/{n_shots_loc})")

        
        print(f"\nQFT nq:{n_qubits_loc}   prob: {prob_loc:.3f} +/- {perr_loc:.3f},  {sbm['backend']} Success: {success_count:.1f}/{n_shots_loc}  job:{md['short_name']}\n")

        print("#s1,backend,taskType,taskParam,qubits,shots,prob,err_prob,cost,expName")
        print(f"#s2,{sbm['backend']},{pmd['task_type']},{inpInt},{n_qubits_loc},{n_shots_loc},{prob_loc:.4f},{perr_loc:.4f},{qa['cost_hqc']},{md['short_name']}")

        return np.array([prob_loc, perr_loc])
 
    #pprint(pmd)
    if pmd['task_type'] == 'ghz':
        bigD['prob'] = postproc_ghz(counts, n_qubits, n_shots)
    elif pmd['task_type'] == 'qft':
        bigD['prob'] = postproc_qft(counts, n_qubits, n_shots, pmd)
    else:
        print(f"Unknown task_type: {pmd.get('task_type')} -- cannot postprocess")
        bigD['prob'] = np.array([0.0, 0.0])
    
    return bigD

#=================================
if __name__ == "__main__":
    args = get_parser()
    
    inpF = args.expName + '.qtuum.npz'
    inpFF = os.path.join(args.inpPath, inpF)
    assert os.path.exists(inpFF), f"Input file not found: {inpFF}"
    expD, expMD = read_data_npz(inpFF, verb=args.verb)
    expMD['verb'] = args.verb
    
    if args.verb > 1: pprint(expMD)

    retrieve_qtuum_job(expMD,expD)
    pprint(expMD['job_qa']) 
    

    if args.verb > 2: pprint(expMD)
    
    #...... WRITE  OUTPUT .........
    if not os.path.exists(args.outPath): os.makedirs(args.outPath)
    outF = os.path.join(args.outPath, expMD['short_name'] + '.meas.npz')
    write_data_npz(expD, outF, expMD)

    print('\n  NO  ./postproc_rb.py  --expName   %s   -p a    -Y\n' % (expMD['short_name']))
