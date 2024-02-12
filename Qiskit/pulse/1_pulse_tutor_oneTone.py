#!/usr/bin/env python3
'''
Basice Pulse exercise 1
==> Measure qubit frequency by performig seweep
==> talk to real HW

This code produces no plots, but the Jupyter notbook on the web works fine

for IBM token go to: https://quantum-computing.ibm.com/account

based on ( takes 1 min to start the server)
https://quantum-computing.ibm.com/lab/files/qiskit-textbook/content/ch-quantum-hardware/calibrating-qubits-pulse.ipynb

'''
import matplotlib.pyplot as plt
import time

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
from qiskit.pulse import Play
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib

from qiskit import IBMQ
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backName='ibmq_armonk'
backend = provider.get_backend(backName)

# verify Pulse is enabled on the backend
backend_config = backend.configuration()
assert backend_config.open_pulse, "Backend doesn't support Pulse"

print('Backend: %s supports Pulse'%backend)

#   - - - - -   access backend config
dt = backend_config.dt # The configuration returns dt in seconds
print('\na) sampling time =%.3f ns'%(dt/ns))

# other HW properties, like qubits frequencies can be obtained via
backend_defaults = backend.defaults()

# We will find the qubit frequency for the following qubit.
qubit = 0
center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
# The default frequency is given in Hz
# warning: this will change in a future release

print('center_frequency_MHz =%.3f'%(center_frequency_Hz/MHz))

# ADD printout code

# - - - - - - Finding the qubit Frequency using a Frequency Sweep
#  sweeping a range of frequencies and looking for signs of absorption using a tool known as a Network Analyzer.
print('\nb) qubit freq sweep')

# scale factor to remove factors of 10 from the data
scale_factor = 1e-14

# We will sweep 40 MHz around the estimated frequency
frequency_span_Hz = 40 * MHz
# in steps of 1 MHz.
frequency_step_Hz = 1 * MHz

# We will sweep 20 MHz above and 20 MHz below the estimated frequency
frequency_min = center_frequency_Hz - frequency_span_Hz / 2
frequency_max = center_frequency_Hz + frequency_span_Hz / 2
# Construct an np array of the frequencies for our experiment
frequencies_GHz = np.arange(frequency_min / GHz, 
                            frequency_max / GHz, 
                            frequency_step_Hz / GHz)

print('The sweep will go from %.1f to %.1f MHz, step=%.1f MHz, shape=%s'%(frequency_min / MHz, frequency_max / MHz,frequency_step_Hz / MHz, frequencies_GHz.shape))

# define drive pulse, which is a Gaussian pulse , trunctaed at +/- 8 std dev
# Drive pulse parameters (us = microseconds)
drive_sigma_us = 0.075                     # This determines the actual width of the gaussian
drive_samples_us = drive_sigma_us*8        # This is a truncating parameter, because gaussians don't have 
                                           # a natural finite length
drive_amp = 0.05

# conversion from phys units to bins, aka ticks
drive_sigma = get_closest_multiple_of_16(drive_sigma_us * us /dt)  # The width of the gaussian in units of dt
drive_samples = get_closest_multiple_of_16(drive_samples_us * us /dt)  # The truncating parameter in units of dt
print('Drive pulse, sigma: %.3f(us)  %d(tick), total len %d(tick), drive_amp=%.3f '%(drive_sigma_us ,drive_sigma, drive_samples,drive_amp))

# Drive pulse samples
drive_pulse = pulse_lib.gaussian(duration=drive_samples,
                sigma=drive_sigma,amp=drive_amp,name='freq_sweep_excitation_pulse')
print('\nM:produced drive_pulse',drive_pulse)
drive_ampls=drive_pulse.samples  # <-- np.array

# check HW constraints map
# Find out which group of qubits need to be acquired with this qubit
meas_map_idx = None
for i, measure_group in enumerate(backend_config.meas_map):
    if qubit in measure_group:
        meas_map_idx = i
        break
assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"
# in this case measure_group has only 1 qubit w/ ID=0 - borring

# - - - - Grab Readout pulse from the calibrated backend
inst_sched_map = backend_defaults.instruction_schedule_map
measure = inst_sched_map.get('measure', qubits=backend_config.meas_map[meas_map_idx])
print('\nM:inspect measure_gate',measure)

# - - - - - - - - assign HW channels
# we specify the channels on which we will apply our pulses.
### Collect the necessary channels
drive_chan = pulse.DriveChannel(qubit)
meas_chan = pulse.MeasureChannel(qubit)
acq_chan = pulse.AcquireChannel(qubit)
print('\nc) HW mapping for my qubit=%d :'%qubit, drive_chan,meas_chan,acq_chan)

# - - - - - SCHEDULE the circuit - - - - -
'''
Now that the pulse parameters have been defined, and we have created the pulse shapes for our experiments, we can proceed to create the pulse schedules.

At each frequency, we will send a drive pulse of that frequency to the qubit and measure immediately after the pulse. The pulse envelopes are independent of frequency, so we will build a reusable `schedule`, and we will specify the drive pulse frequency with a frequency configuration array.
'''
# Create the base schedule
# Start with drive pulse acting on the drive channel
schedule = pulse.Schedule(name='Frequency sweep')
schedule += Play(drive_pulse, drive_chan)
# The left shift `<<` is special syntax meaning to shift the start time of the schedule by some duration
schedule += measure << schedule.duration

# Create the frequency settings for the sweep (MUST BE IN HZ)
frequencies_Hz = frequencies_GHz*GHz
schedule_frequencies = [{drive_chan: freq} for freq in frequencies_Hz]

# - - - - - DRAW Pulse Envelope- - - - - -
# As a sanity check, it's always a good idea to look at the pulse schedule.
plt.figure(figsize=(8, 6))
#ax = plt.subplot(1, 1, 1)

# doc: https://qiskit.org/documentation/stubs/qiskit.pulse.Schedule.html#qiskit.pulse.Schedule.draw
print('plot args:',schedule.draw.__code__.co_argcount ,schedule.draw.__code__.co_varnames)

schedule.draw(label=True)
print('sched-draw takes 5 min to display ...')

'''
 - - - - The mosur interesting section
We assemble the schedules and schedule_frequencies above into a program object, called a Qobj, that can be sent to the quantum device. We request that each schedule (each point in our frequency sweep) is repeated num_shots_per_frequency times in order to get a good estimate of the qubit response.

We also specify measurement settings. meas_level=0 returns raw data (an array of complex values per shot), meas_level=1 returns kerneled data (one complex value per shot), and meas_level=2 returns classified data (a 0 or 1 bit per shot). We choose meas_level=1 to replicate what we would be working with if we were in the lab, and hadn't yet calibrated the discriminator to classify 0s and 1s. We ask for the 'avg' of the results, rather than each shot individually.
'''


from qiskit import assemble

num_shots_per_frequency = 20
frequency_sweep_program = assemble(schedule,
                                   backend=backend, 
                                   meas_level=1,
                                   meas_return='avg',
                                   shots=num_shots_per_frequency,
                                   schedule_los=schedule_frequencies)
job = backend.run(frequency_sweep_program)
print('M:jobId=',job.job_id(), 'shots/freq=',num_shots_per_frequency )
from qiskit.tools.monitor import job_monitor
job_monitor(job)

print('M:retrieve results')
T0=time.time()
#Once the job is run, the results can be retrieved using:
frequency_sweep_results = job.result(timeout=120) # timeout parameter set to 120 seconds
print('M:retrieve done, elaT=%.1f sec'%(time.time()-T0))

sweep_values=[]
for i in range(len(frequency_sweep_results.results)):
    # Get the results from the ith experiment
    res = frequency_sweep_results.get_memory(i)*scale_factor
    # Get the results for `qubit` from this experiment
    print('IQ-pairs:',i,res[qubit])
    sweep_values.append(res[qubit])

print('M:plotting')
plt.scatter(frequencies_GHz, np.real(sweep_values), color='black') # plot real part of sweep values
plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
plt.xlabel("Frequency [GHz]")
plt.ylabel("Measured signal [a.u.]")


plt.show()


