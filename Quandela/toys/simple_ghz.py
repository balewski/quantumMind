#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
Construct GHZ state uising Heralded (Knill) CNOT gate

'''

num_qubit=2  # <=== change  GHZ state size here

import perceval as pcvl
from perceval.algorithm import Sampler
print('perceval ver:',pcvl.__version__)

cnot = pcvl.catalog["heralded cnot"].build_processor()
num_mode=2*num_qubit
# Aubaert:  each heralded cnots brings two added photons for the heralds. As such, your min_detected_photons_filter is not high enough. It should be num_qubit + 2 * (num_qubit - 1).
min_photon=num_qubit + 2 * (num_qubit - 1)

proc = pcvl.Processor("SLOS",num_mode)
proc.min_detected_photons_filter(min_photon)
proc.add(0, pcvl.BS.H())
for j in range(1,num_qubit):
    proc.add(2*j-2, cnot)
pcvl.pdisplay(proc)
shots = 1000
sampler = Sampler(proc, max_shots_per_call=shots)    

dualSt=pcvl.BasicState([1,0]*num_qubit) # bitSt='00'
proc.with_input(dualSt )
resD=sampler.sample_count()  
photC=resD['results'] # number of photons in each mode.
print('GHZ num_qubit=%d, measured:%s'%(num_qubit,photC))
print('transmission:',resD['physical_perf'])
