#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
'''
import perceval as pcvl
from perceval.algorithm import Sampler
cnotH = pcvl.catalog["heralded cnot"].build_processor()

from time import time,sleep

def monitor_async_job(remote_job, numSec=10):
    T0=time()
    i=0
    while True:
        jstat=remote_job.status()
        elaT=time()-T0
        print('M:i=%d  status=%s, elaT=%.1f sec'%(i,jstat,elaT))
        if jstat=='SUCCESS': break
        if jstat=='ERROR': exit(99)
        i+=1; sleep(numSec)
    return  

#=================================
#  M A I N 
#=================================
if __name__ == "__main__":

    num_qubit=2
    nMode=2*num_qubit
    minPhoton=num_qubit  # Reduced minimum photon requirement
    isIdeal=True
    
    if isIdeal:
        proc = pcvl.Processor("SLOS",nMode)
    else:
        proc = pcvl.RemoteProcessor("sim:ascella",m=nMode)

    # Set up the circuit: H gate + CNOT to create Bell state
    proc.add(0, pcvl.BS.H())  # Hadamard on mode 0
    proc.add(0, cnotH)        # CNOT with heralding

    # Set the input state - 1 photon in mode 0, 1 photon in mode 2
    input_state = pcvl.BasicState([1,0,1,0])
    proc.with_input(input_state)

    # Apply the minimum photon filter after setting up the circuit
    proc.min_detected_photons_filter(minPhoton)

    pcvl.pdisplay(proc)
    print('M: input_state=%s, min_photons=%d' % (str(input_state), minPhoton),proc.name)
  
    shots=1000_000_000  # Reduced for testing
    sampler = Sampler(proc, max_shots_per_call=shots)
    if isIdeal:
        resD=sampler.sample_count(shots)
    else:
        job = sampler.sample_count.execute_async(shots)
        monitor_async_job(job)
        resD=job.get_results()
    #print(resD)

    # Handle different performance metrics based on simulator type
    if isIdeal:
        # Local/ideal simulator
        print('perf: global=%.6f' % resD['global_perf'])
    else:
        # Remote/noisy simulator
        print('perf: physical=%.6e, logical=%.6f' % (resD['physical_perf'], resD['logical_perf']))

    print('counts:', resD['results'])
    print('shots=%2g' % shots,proc.name) 

