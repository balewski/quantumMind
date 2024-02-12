#!/usr/bin/env python3

'''
It uses all available degrees of freedom (except global phase scheduling), namely:
*Global detuning schedule (aka time dependence) may vary between +/- 124 (rad/us).
*Local detune has only 1 common schedule, and  the magnitude can be only positive [0, 62 ] (rad/us)
*You can only change the  overall scaling of local detuning per atom by a factor in the range [0,1], the time dependence is common
The local & global detuning are added for every time bin, so effectively you can have also negative local detuning, but there are some restriction on the amplitudes,
'''

from bloqade import start
import numpy as np

from toolbox.Util_bloqade import invert_keys, append_state_energy


# Define relevant parameters
atomPos=np.array([(0,0),(5,0),(10,0)]) # (um)
register = start.add_position(atomPos)
print(register)

durations = [0.8, "sweep_time", 0.8]  # (us)
omega_max=15.7  # (rad/us)
delta_glob_max=124.  # (rad/us)
delta_loc_max=62.  # (rad/us)

# add local detuning
atomLab=[0,1,2]  # atom names labeled by their placement order. Counting from 0?
atomScale=[0.7,0.5,0.3]  # (range [0,1])  those are  modulations for the common local detune
nAtom=len(atomLab)
detuneLast=[ delta_glob_max + atomScale[i]*delta_loc_max  for i in range(nAtom) ]
print('detuneLast (rad/us):',detuneLast)

program = (
    register
    .rydberg.rabi.amplitude.uniform.piecewise_linear(
        durations, values=[0.0, "rabi_drive", "rabi_drive", 0.0])
    .rydberg.detuning.uniform.piecewise_linear(
        durations, values=["detune_min","detune_min","detune_max","detune_max"])
    .rydberg.detuning.location(atomLab,atomScale).piecewise_linear(
        durations, values=[0.,0.,"detune_local","detune_local"])
    .assign(sweep_time=1.2,rabi_drive=omega_max,
            detune_min=-delta_glob_max, detune_max=delta_glob_max, detune_local=delta_loc_max)
)

#1print(program)

emu_batch = program.bloqade.python().run(100)
report = emu_batch.report()

counts=invert_keys(report.counts)[0]   # '0'=ground,'1'=rydberg; take only 1st circuit
print('\nCounts:',counts)

# compute energy for masured states
append_state_energy(counts,atomPos,detuneLast,verb=1)
print('\nCounts+energy:',counts)
