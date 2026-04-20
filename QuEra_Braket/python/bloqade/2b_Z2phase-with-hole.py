#!/usr/bin/env python3
# coding: utf-8

# create Z2-phase with a hole

from bloqade.atom_arrangement import Chain
import numpy as np
import os, time

from toolbox.Util_bloqade import invert_keys, append_state_energy

shots=300
# Define relevant parameters for the lattice geometry and pulse schedule
nAtom = 9 
lattice_const = 6.1  # (um)
atomPos=[ (0,lattice_const*i) for i in range(nAtom) ]
atomPos=np.array(atomPos)
register=Chain(nAtom, lattice_const)
print(register)

# Define Rabi amplitude and detuning values.
rabi_amplitude = [0.0, 15., 15., 0.0]
delta_glob = [-16, -16, 16, 16]
durations = [0.5, "sweep_time", 0.5]
detuneLast=[ delta_glob[-1]  for i in range(nAtom) ]
print('detuneLast (rad/us):',detuneLast)

prog1 = (
    register
    .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, rabi_amplitude)
    .detuning.uniform.piecewise_linear(durations, delta_glob)
)


job1 = prog1.assign(sweep_time=3.0)
#print(program)
t0=time.time(); print('\nrun nAtom=%d  shots=%d  program1 ...'%(nAtom,shots))
emu1 = job1.bloqade.python().run(shots)
print('done run elaT=%.1f sec'%(time.time()-t0))
report1 = emu1.report()

counts1=invert_keys(report1.counts)[0]   #  is '1'=rydberg
print('Counts1:',counts1)
append_state_energy(counts1,atomPos,detuneLast,verb=1)


# use detunning to suppress 1-state on atom #2 counting form 0 ,the  goal state is: 100010101 


delta_glob2 = [-16, -16, -16, -16]
delta_loc2 = [0., 0.,  32., 32]
atomLab=[i for i in range(nAtom) ]
atomScale=[1. for i in range(nAtom) ]
for i in range(3): atomScale[1+i]=0.

detuneLast2=[ delta_glob2[-1] + atomScale[i]*delta_loc2[-1]  for i in range(nAtom) ]
print('detuneLast2 (rad/us):',detuneLast2)

prog2 = (
    register
    .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, rabi_amplitude)
    .detuning.uniform.piecewise_linear(durations, delta_glob2)
    .location(atomLab,atomScale).piecewise_linear(durations, values=delta_loc2)
)


print('atoms subject to detune:',atomScale)


job2 = prog2.assign(sweep_time=3.0)
t0=time.time(); print('\nrun nAtom=%d  program2 ...'%nAtom)
emu2 = job2.bloqade.python().run(shots)
print('done run elaT=%.1f sec'%(time.time()-t0))
report2 = emu2.report()

counts2=invert_keys(report2.counts)[0]    #  is '1'=rydberg
print('Counts2:',counts2)
append_state_energy(counts2,atomPos,detuneLast2,verb=1)



