#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Post-process Guppy zoo measurement counts stored by `retrieve_qtuum_zoo.py`.

This script reads packed counts from `out/meas/*.meas.npz`,
computes task-specific figures of merit for GHZ or QFT runs,
stores the result in `expD['prob']`, and writes `out/post/*.post.npz`.

Usage:
  ./postproc_zoo.py --expName ideal_3fbad8
'''

import os
from pprint import pprint
import numpy as np

from toolbox.Util_NumpyIOv1 import write_data_npz, read_data_npz
from toolbox.Util_CountsPacker import unpack_counts_from_npz


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3, 4], default=1, dest='verb')
    parser.add_argument("--basePath", default='out', help="base directory for measurement and postprocessing files")
    parser.add_argument('-e', "--expName", default=None, help='experiment name to postprocess')

    args = parser.parse_args()
    args.dataPath = os.path.join(args.basePath, 'meas')
    args.outPath = os.path.join(args.basePath, 'post')

    print('myArg-program:', parser.prog)
    for arg in vars(args):
        print('myArg:', arg, getattr(args, arg))

    assert os.path.exists(args.dataPath), f"missing dataPath: {args.dataPath}"
    assert os.path.exists(args.outPath), f"missing outPath: {args.outPath}"
    return args


#...!...!....................
def postproc_ghz(counts_dict, n_qubits, n_shots, md):
    state0 = '0' * n_qubits
    state1 = '1' * n_qubits
    prob = (counts_dict.get(state0, 0) + counts_dict.get(state1, 0)) / n_shots
    err_prob = np.sqrt(prob * (1 - prob) / n_shots)
    print(f"GHZ nq:{n_qubits} Fidelity : {prob:.4f} +/- {err_prob:.4f}  fail: {1 - prob:.4f}  job:{md['short_name']}\n")
    return np.array([prob, err_prob])


#...!...!....................
def postproc_qft(counts_dict, n_qubits, n_shots, md):
    pmd = md['payload']
    sbm = md['submit']
    qa = md['job_qa']

    inp_int = int(pmd['task_param'][0])
    target_str = f"{inp_int:0{n_qubits}b}"
    success_count = counts_dict.get(target_str, 0)
    prob = success_count / n_shots
    err_prob = np.sqrt(prob * (1 - prob) / n_shots)

    wrong_total_counts = n_shots - success_count
    any_states = len(counts_dict)
    wrong_states = any_states - 1
    wrong_prob = wrong_total_counts / n_shots
    print(f"Incorrect states: {wrong_states} of {any_states}, total probability: {wrong_prob:.4f} ({wrong_total_counts:.1f}/{n_shots})")

    print(f"\nQFT nq:{n_qubits}   prob: {prob:.3f} +/- {err_prob:.3f},  {sbm['backend']} Success: {success_count:.1f}/{n_shots}  job:{md['short_name']}\n")

    print("#s1,backend,taskType,taskParam,qubits,shots,prob,err_prob,cost,expName")
    print(f"#s2,{sbm['backend']},{pmd['task_type']},{inp_int},{n_qubits},{n_shots},{prob:.4f},{err_prob:.4f},{qa['cost_hqc']},{md['short_name']}")

    return np.array([prob, err_prob])


#...!...!....................
def postproc_zoo(expD, expMD):
    countsL = unpack_counts_from_npz(expD)
    if args.verb > 1:
        pprint(countsL)

    counts = countsL[0]
    n_qubits = expMD['payload']['num_qubits']
    n_shots = expMD['submit']['num_shots']
    task_type = expMD['payload']['task_type']

    if task_type == 'ghz':
        expD['prob'] = postproc_ghz(counts, n_qubits, n_shots, expMD)
    elif task_type == 'qft':
        expD['prob'] = postproc_qft(counts, n_qubits, n_shots, expMD)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


#=================================
if __name__ == "__main__":
    args = get_parser()
    np.set_printoptions(precision=3)

    inpF = args.expName + '.meas.npz'
    inpFF = os.path.join(args.dataPath, inpF)
    assert os.path.exists(inpFF), f"Input file not found: {inpFF}"

    expD, expMD = read_data_npz(inpFF, verb=args.verb)

    if args.verb >= 2:
        print('M:expMD:')
        pprint(expMD)

    postproc_zoo(expD, expMD)

    outF = os.path.join(args.outPath, expMD['short_name'] + '.post.npz')
    write_data_npz(expD, outF, expMD, verb=1)
    print('M:done')
