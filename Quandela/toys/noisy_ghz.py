#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
Construct GHZ state uising Heralded (Knill) CNOT gate
Use local noisy  Sampler

'''
import perceval as pcvl
from perceval.algorithm import Sampler
import numpy as np
print('perceval ver:',pcvl.__version__)

cnot = pcvl.catalog["heralded cnot"].build_processor()

source = pcvl.Source(emission_probability=0.95, multiphoton_component=0.01)
#source=None  # activate to disable noise

num_qubit=3  # <=== change  GHZ state size here
num_mode=2*num_qubit

# Set detected photons filter and circuit
proc = pcvl.Processor("SLOS",num_mode,source)
proc.min_detected_photons_filter(num_qubit)
proc.add(0, pcvl.BS.H())
print('moi:',proc._n_moi) #: int = 0  # Number of modes of interest (moi)
for j in range(num_qubit-1):
    proc.add(2*j, cnot)
pcvl.pdisplay(proc)

# Run the simulation
shots = 10_000
sampler = Sampler(proc, max_shots_per_call=shots)    

st0=pcvl.BasicState([1,0]*num_qubit) # all qubits in 0-state
st1=pcvl.BasicState([0,1]*num_qubit) # all qubits in 1-state
proc.with_input(st0 )
resD=sampler.sample_count()  
photC=resD['results'] # number of photons in each mode.

n0=0
n1=0
k=0
for phSt, count in photC.items():
    tag=' '
    if phSt==st0:  
        n0=count
        tag='*'
    if phSt==st1:  
        n1=count
        tag='*'
    if k<15 or tag=='*':
        print('%s meas phSt %s,  count %d'%(tag,phSt,count))
    else: print('.',end='')
    k+=1

print('\ntransmission: %.3f  num final states: %d'%(resD['physical_perf'], k))
print('0s prob: %.4f'% (n0/shots)) 
print('1s prob: %.4f'% (n1/shots))
print(' num sigma: |no-n1|/sqrt(n0+n1)=%.3f '%( abs(n0-n1)/np.sqrt(n0+n1))) 
print('asymmetry: n0-n1/n0+n1 =%.4f'%( (n0-n1)/(n0+n1)))
print('fidelity: n0+n1/ns=%.3f'%((n0+n1)/shots))
