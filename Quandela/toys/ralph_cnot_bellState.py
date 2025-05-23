#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
Construct bell-state using  

https://perceval.quandela.net/docs/v0.12/notebooks/Tomography_walkthrough.html
The Heralded (Knill) CNOT gate

Use:  local noisy  Sampler

init dualRail: |1,0,1,0> --> bitSt: 00
meas bitSt: 11 500
meas bitSt: 01 2
meas bitSt: 00 490
meas bitSt: 10 8
transmission: 0.063

'''
import numpy as np
from pprint import pprint
import perceval as pcvl
from perceval.algorithm import Sampler
print('perceval ver:',pcvl.__version__)

from toolbox.Util_Quandela  import dualRailState_to_bitStr

cnot = pcvl.catalog["heralded cnot"].build_processor()
nMode=4

# define noise level
source = pcvl.Source(emission_probability=0.99, multiphoton_component=0.01)

# Set detected photons filter and circuit
proc = pcvl.Processor("SLOS",nMode,source)

proc.min_detected_photons_filter(2)
proc.add(0, pcvl.BS.H())
proc.add(0, cnot)
pcvl.pdisplay(proc)

# Run the simulation
shots = 10_000
sampler = Sampler(proc, max_shots_per_call=shots)    

fockStt=pcvl.BasicState([1,0,1,0]) # bitSt='00'
proc.with_input(fockStt )
resD=sampler.sample_count()  
photC=resD['results'] # number of photons in each mode.
print('\ninit dualRail:',fockStt,'--> bitSt:',dualRailState_to_bitStr(fockStt))
for phStt, count in photC.items():
    bitStr=dualRailState_to_bitStr(phStt)
    print('meas bitStr:',bitStr,count)
print('transmission:',resD['physical_perf'])

