#!/usr/bin/env python3
# coding: utf-8

# create 2-atom states: 01,10,11 with not equal probability
# use only global detune
''' from Milo:
If you are infinitely slow (ideal adiabatic), you will end up in the manifold spanned by these 3 states, but in which state exactly depends on how quantum effects of Omega act there - Omega will split this manifold as in degenerate perturbation theory. 
'''


from bloqade.atom_arrangement import Chain
import numpy as np
import os, time

from toolbox.Util_bloqade import invert_keys, append_state_energy

shots=10000
# Define relevant parameters for the lattice geometry and pulse schedule
nAtom = 2 
lattice_const = 4.0  # (um)
atomPos=[ (0,lattice_const*i) for i in range(nAtom) ]
atomPos=np.array(atomPos)
register=Chain(nAtom, lattice_const)
print(register)

# Define Rabi amplitude and detuning values.
rabi_amplitude = [0.0, 15., 15., 0.0]
delta_glob = [-20, -20, 1323, 1323]
durations = ["ramp_time", "sweep_time", "ramp_time"]
detuneLast=[ delta_glob[-1]  for i in range(nAtom) ]
print('detuneLast (rad/us):',detuneLast)
#print('atomPos (um)',atomPos)
append_state_energy(['00','01','10','11'],atomPos,detuneLast,verb=1)

prog = (
    register
    .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, rabi_amplitude)
    .detuning.uniform.piecewise_linear(durations, delta_glob)
)

# vary duration of the sweep time
for tsweep in [3,6,10,30,50] :
    job = prog.assign(ramp_time=0.5,sweep_time=tsweep)
    t0=time.time(); print('\nrun nAtom=%d  shots=%d  tsweep=%.1f us ...'%(nAtom,shots,tsweep))
    emu = job.bloqade.python().run(shots)
    print('  done run elaT=%.1f sec'%(time.time()-t0))
    report = emu.report()

    counts=invert_keys(report.counts)[0]   #  is '1'=rydberg
    #print('Counts:',counts)

    append_state_energy(counts,atomPos,detuneLast,verb=0)
    for key,val in counts.items():        
        m,e,_,_=val
        print('key=%s mshot=%d ene=%.1f MHz'%(key,m,e))
    
