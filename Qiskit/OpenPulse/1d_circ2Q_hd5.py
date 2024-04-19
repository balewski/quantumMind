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
from pulseIO_util import qiskitSched_dump,qiskitSched_unpack
from h5io_util import write_data_hdf5, read_data_hdf5
from pprint import pprint
import json

import qiskit
from qiskit import QuantumCircuit, transpile,schedule
from qiskit.test.mock import FakeOpenPulse2Q

#=================================
#=================================
#  M A I N 
#=================================
#=================================
print(qiskit.__qiskit_version__)

backend = FakeOpenPulse2Q()
print('Backend: %s '%backend)
#1assert backend_config.open_pulse, "Backend doesn't support Pulse"

# quantum circuit to make a Bell state 
bell = QuantumCircuit(2, 2)
bell.h(0)
bell.t(1)
bell.delay(4) # unit is dt
bell.cx(0,1)
bell.delay(4) # unit is dt
bell.x(0)
bell.h(1)
print('circuit:'); print(bell)

qc=transpile(bell,backend)
print('transpiled:'); print(qc)
qc.draw(output='mpl')

sched_circ=schedule(qc,backend)
#print(type(sched_circ),'plot args:',sched_circ.draw.__code__.co_argcount ,sched_circ.draw.__code__.co_varnames)

fig, ax = plt.subplots(figsize=(14, 5))
sched_circ.draw('IQXDebugging', axis = ax, show_waveform_info = True,plot_range=[0, 60])
ax.grid(axis='x', linestyle='--')
#plt.show()

# schedule += Play(drive_pulse, drive_chan)
print('scheduled:',len(sched_circ))

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
outF='circ1.schedule.h5'
write_data_hdf5(waveformD,outF)

# convert back
blob=read_data_hdf5(outF)
stepR = json.loads(blob['schedule_steps'])
print('M:stepL recovered', type(stepR), len(stepR)); pprint(stepR)
name='5_cr90m_u0'
wave= blob[name]
print('M: recovered', name,wave.shape,wave.dtype)
    
#plt.show()

