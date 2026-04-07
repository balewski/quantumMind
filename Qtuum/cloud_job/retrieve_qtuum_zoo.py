#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Retrieve Guppy zoo results from a Quantinuum Nexus job.

This script loads submission metadata from `out/jobs/*.qtuum.npz`,
downloads raw result counts from Nexus, records job QA metadata,
and writes packed measurement counts to `out/meas/*.meas.npz`.

 Use case:
 ./retrieve_qtuum_zoo.py --expName helios_3fbad8
'''

import os, sys
from pprint import pprint
from time import time, sleep
import json
import pytz
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qnexus.models").setLevel(logging.ERROR)
import qnexus as qnx

from qnexus.models.references import ExecuteJobRef

from toolbox.Util_NumpyIOv1 import write_data_npz, read_data_npz
from toolbox.Util_IOfunc import dateT2Str
from toolbox.Util_CountsPacker import pack_counts_for_npz


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase output verbosity", default=1)
    parser.add_argument("--basePath",default='out',help="base directory for jobs and measurements")
    parser.add_argument('-e',"--expName",  default=None, help='experiment name created during submission')
    
    parser.add_argument( "--cancelJob", action='store_true', default=False, help="reserved flag; not used by the current retrieval flow")

    args = parser.parse_args()
    args.inpPath=os.path.join(args.basePath,'jobs')
    args.outPath=os.path.join(args.basePath,'meas')
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def retrieve_qtuum_job(md,bigD):
    #pprint(md)
    sbm=md['submit']
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
        'num_shots': sbm['num_shots'],
        'cost_hqc': getattr(jobStatus, 'cost', -1.1),
        'job_id': ref_exec.id,
    }
    qa['status']=str(stat)
    qa['num_circ']=nCirc

    dt = jobStatus.completed_time - jobStatus.running_time
    qa['run_sec'] = dt.total_seconds()

    cal_tz = pytz.timezone("America/Los_Angeles")
    dt_pt = jobStatus.running_time.astimezone(cal_tz)
    qa['timestamp_running'] = dateT2Str(dt_pt.timetuple())

    countsL=[None]*nCirc
    for ic in range(nCirc):
        result_nx = results[ic].download_result()
        counts_nx = result_nx.collated_counts()
        # Extract bitstrings. Keys are tuples of (tag, bitstring) pairs.
        bitstring_dict = {"".join(tag_res[1] for tag_res in key)[::-1]: val for key, val in counts_nx.items()}
        countsL[ic]=bitstring_dict
        #print('\nis=%d  res:'%(ic)); pprint(bitstring_dict)

    #print('job QA'); pprint(qa)
    md['job_qa']=qa

    return countsL

#=================================
if __name__ == "__main__":
    args = get_parser()
    
    inpF = args.expName + '.qtuum.npz'
    inpFF = os.path.join(args.inpPath, inpF)
    assert os.path.exists(inpFF), f"Input file not found: {inpFF}"
    expD, expMD = read_data_npz(inpFF, verb=args.verb)
    expMD['verb'] = args.verb
    
    if args.verb > 1: pprint(expMD)

    countsL=retrieve_qtuum_job(expMD,expD)
    bigD=pack_counts_for_npz(countsL)
    print('job QA'); pprint(expMD['job_qa']) 
    
    if args.verb > 2: pprint(expMD)
    
    #...... WRITE  OUTPUT .........
    if not os.path.exists(args.outPath): os.makedirs(args.outPath)
    outF = os.path.join(args.outPath, expMD['short_name'] + '.meas.npz')
    write_data_npz(bigD, outF, expMD)

    print('\n   ./postproc_zoo.py  --expName   %s\n' % (expMD['short_name']))
