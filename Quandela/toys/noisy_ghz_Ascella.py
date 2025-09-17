#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

'''
from noisy_bell import monitor_async_job

import perceval as pcvl
from perceval.algorithm import Sampler
cnotH = pcvl.catalog["heralded cnot"].build_processor()
cnotP = pcvl.catalog['postprocessed cnot'].build_processor()


#=================================
#  M A I N 
#=================================
if __name__ == "__main__":

    num_qubit=3
    nMode=2*num_qubit
    minPhoton=num_qubit 
    isIdeal=True
    
    if isIdeal:
        proc = pcvl.Processor("SLOS",nMode)
    else:
        proc = pcvl.RemoteProcessor("sim:ascella",m=nMode)
    proc.min_detected_photons_filter(minPhoton)
    proc.add(0, pcvl.BS.H())
    proc.add(0, cnotH)
    proc.add(2, cnotP)
    proc.with_input(pcvl.BasicState([1,0,1,0,1,0]))
    pcvl.pdisplay(proc)
    print('M: proc:',proc.name)
    
    shots=2_000_000_000
    sampler = Sampler(proc, max_shots_per_call=shots)
    if isIdeal:
        resD=sampler.sample_count(shots)
    else:
        job = sampler.sample_count.execute_async(shots)
        monitor_async_job(job)
        resD=job.get_results()

    #Handle different performance metrics based on simulator type
    if isIdeal:
        # Local/ideal simulator
        print('perf: global=%.6f' % resD['global_perf'])
    else:
        # Remote/noisy simulator
        print('perf: physical=%.6e, logical=%.6f' % (resD['physical_perf'], resD['logical_perf']))

    print('counts:', resD['results'])
    print('shots=%2g' % shots,proc.name)


