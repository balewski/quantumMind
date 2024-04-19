#!/usr/bin/env python3
'''
Basice Pulse exercise
export pulse as HD5

for IBM token go to: https://quantum-computing.ibm.com/account

based on 
https://quantum-computing.ibm.com/lab/files/qiskit-textbook/content/ch-quantum-hardware/calibrating-qubits-pulse.ipynb

'''

from h5io_util import write_data_hdf5, read_data_hdf5
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
from qiskit.pulse import Play
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib


from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
backName="ibm_kyoto"
#backName="ibm_hanoi"
backend = service.get_backend(backName)


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
qubit = 22
center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
# The default frequency is given in Hz
# warning: this will change in a future release
print(backend, 'Q:%d center_frequency_MHz =%.3f'%(qubit,center_frequency_Hz/MHz))

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
drv_ampls=drive_pulse.samples  # <-- np.array
print('drive_ampl:',drv_ampls.shape, drv_ampls.dtype,'\n',drv_ampls[::100])

outD={'drive_amplitude':drv_ampls, 'time_step':dt}
outF='drv_pulse.h5'
write_data_hdf5(outD,outF,verb=1)

xx=read_data_hdf5(outF,verb=1)
dt=xx['time_step']
print('end %s %.3g'%( type(dt),dt))

#.....  run circuit on the hardware .....
