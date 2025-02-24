#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
Verify truth table for Rolph-CNOT contstruct
Use local noisy  Sampler
https://arxiv.org/pdf/quant-ph/0112088

Correct truth table
	00	01	10	11
00	1	0	0	0
01	0	1	0	0
10	0	0	0	1
11	0	0	1	0

Noisy simulation
  ['00', '01', '10', '11']
00 [992   0   8   0]
01 [  0 996   0   4]
10 [  0   0  13 987]
11 [  0   0 984  16]

'''
import numpy as np
from pprint import pprint
import perceval as pcvl
from perceval.algorithm import Sampler
print('perceval ver:',pcvl.__version__)

from toolbox.Util_Quandela  import dualRailState_to_bitstring, bitstring_to_dualRailState, dualrail_to_bitstring, bitstring_to_dualrail

# create Ralph CNOT Gate as a processor - it does the rejection logic inside (?)
pRCnot = pcvl.catalog['postprocessed cnot'].build_processor()
#pcvl.pdisplay(pRCnot, recursive=True)

bitStateL=['00','01','10','11']
nCirc=len(bitStateL)

# define noise level
source = pcvl.Source(emission_probability=0.25, multiphoton_component=0.01)

# Set detected photons filter and circuit
proc = pcvl.Processor("SLOS",4,source)
proc.min_detected_photons_filter(2)
proc.add(0, pRCnot)
pcvl.pdisplay(proc)

# Run the simulation
shots = 1000
sampler = Sampler(proc, max_shots_per_call=shots)    

print('\n ',bitStateL)
for j in range(nCirc):
    bitSt=bitStateL[j]
    dualSt=bitstring_to_dualRailState(bitSt )
    proc.with_input(dualSt )
    resD=sampler.sample_count()  
    photC=resD['results'] # number of photons in each mode.
    #print('\ninit:',bitSt,'meas:',photC)
    outCnt=np.zeros(nCirc,dtype=int)
    for phSt, count in photC.items():
        #print("photon state:", phSt, "Count:", count)
        quSt=dualrail_to_bitstring(phSt)
        i= bitStateL.index(quSt)
        #print(quSt,i)
        outCnt[i]=count
    print(bitSt, outCnt)
    
print('transmission:',resD['physical_perf'])

