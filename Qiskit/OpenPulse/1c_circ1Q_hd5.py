#!/usr/bin/env python3
'''
Basice Pulse exercise
export pulse as HD5

for IBM token go to: https://quantum-computing.ibm.com/account

based on 
https://quantum-computing.ibm.com/lab/files/qiskit-textbook/content/ch-quantum-hardware/calibrating-qubits-pulse.ipynb

'''

import matplotlib.pyplot as plt
import numpy as np

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds
# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)

from qiskit import pulse  # This is where we access all of our Pulse features!
from qiskit import QuantumCircuit, transpile,schedule
from qiskit.pulse import Play

from qiskit.pulse.instructions.phase import ShiftPhase
from qiskit.pulse.instructions.delay import Delay
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib

from qiskit import IBMQ
import qiskit

from pulseIO_util import qiskitSched_dump,qiskitSched_unpack
from h5io_util import write_data_hdf5, read_data_hdf5
from pprint import pprint
import json

#=================================
#=================================
#  M A I N 
#=================================
#=================================
print(qiskit.__qiskit_version__)

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backName='ibmq_armonk'
backend = provider.get_backend(backName)

# verify Pulse is enabled on the backend
backend_config = backend.configuration()
assert backend_config.open_pulse, "Backend doesn't support Pulse"

print('Backend: %s supports Pulse'%backend)
#   - - - - -   access backend config
time_step = backend_config.dt # The configuration returns dt in seconds
print('backend time_step=%.3f ns'%(time_step/ns))


# quantum circuit to make a Bell state 
bell = QuantumCircuit(1)
bell.u(0.4,0.6,0.9,0)
bell.delay(200)
bell.x(0)
bell.s(0)
bell.sx(0)
bell.y(0)
circ = bell
circ.draw(output='mpl')

qc=transpile(circ,backend)
qc.draw(output='mpl')
print('transpiled:'); print(qc)

# schedule += Play(drive_pulse, drive_chan)
sched_circ=schedule(qc,backend)

fig, ax = plt.subplots(figsize=(14, 5))
sched_circ.draw('IQXDebugging', axis = ax, show_waveform_info = True,plot_range=[0, 60])
ax.grid(axis='x', linestyle='--')


print('scheduled:'); print(len(sched_circ))
#1qiskitSched_dump(sched_circ)
stepL,waveformD,=qiskitSched_unpack(sched_circ)
print('\nM: stepL=%d, waveformD=%d'%(len(stepL),len(waveformD)))
pprint(stepL)
for x in waveformD:
    print('waveform',x, waveformD[x].shape,waveformD[x].dtype)

# convert stepL to a single string
stepLtxt=json.dumps(stepL)
print('M:stepLtxt', type(stepLtxt), len(stepLtxt), stepLtxt)
# add steps to output
waveformD['schedule_steps']=stepLtxt
waveformD['time_step']=time_step
outF='circ1.schedule.h5'
write_data_hdf5(waveformD,outF)

plt.show()
