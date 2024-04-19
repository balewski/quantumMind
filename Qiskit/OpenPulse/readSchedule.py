#!/usr/bin/env python3
from h5io_util import write_data_hdf5, read_data_hdf5
import json
from pprint import pprint

outF='circ1.schedule.h5'

blob=read_data_hdf5(outF)
time_step=blob.pop('time_step')[0]
stepL = json.loads(blob.pop('schedule_steps'))
print('M:stepL recovered', type(stepL), len(stepL),'time, _step=',time_step)
pprint(stepL)
waveformD=blob

for x in waveformD:
    print('waveform',x, waveformD[x].shape,waveformD[x].dtype)

print('dump::10',x,waveformD[x][::10])
