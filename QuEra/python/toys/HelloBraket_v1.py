#!/usr/bin/env python3
# coding: utf-8


# ### Goal 
# 
# To say hello to the world of neutral atoms
# Let's start off with the 1D case.   To achieve our goal, we first need to define our Rydberg-atom quantum computer register. We arrange them in a line with a separation of 6.1μm between each pair of atoms. 


from braket.ahs.atom_arrangement import AtomArrangement
import numpy as np
from pprint import pprint as pp
from quera_ahs_utils.plotting import show_register
np.set_printoptions(precision=4) 

a = 6.1e-6  # meters
N_atoms = 11

register = AtomArrangement()
for i in range(N_atoms):
    register.add([0.0, i*a])

print('\nM:dump coordinates',register)

xL=register.coordinate_list(0)
yL=register.coordinate_list(1)

for x,y in zip(xL,yL):
    print(x,y)


# ### Hamiltonian
# 
# The next component we need to specify is the Hamiltonian. 


from braket.ahs.hamiltonian import Hamiltonian

H = Hamiltonian()

from quera_ahs_utils.plotting import show_global_drive 
from quera_ahs_utils.drive import get_drive

omega_min = 0       
omega_max = 2.5e6 * 2 * np.pi
detuning_min = -9e6 * 2 * np.pi
detuning_max = 7e6 * 2 * np.pi

time_max = 4e-6
time_ramp = 0.15*time_max

time_points = [0, time_ramp, time_max - time_ramp, time_max]
omega_values = [omega_min, omega_max, omega_max, omega_min]
detuning_values = [detuning_min, detuning_min, detuning_max, detuning_max]
phase_values = [0, 0, 0, 0]

drive = get_drive(time_points, omega_values, detuning_values, phase_values)
H += drive

print('\nM:dump drive Amplitude :\n',drive.amplitude.time_series.times(), drive.amplitude.time_series.values())


# ### Defining the program (1D case)   MIS
#

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation

ahs_program = AnalogHamiltonianSimulation(
    register=register, 
    hamiltonian=H
)

print('\nM:dump Schrodinger problem:')
circD=ahs_program.to_ir().dict()
pp(circD)

# ### Simulation on classical hardware
# 


from braket.devices import LocalSimulator

classical_device = LocalSimulator("braket_ahs")

nshots = 20
print('\nM:localSimu nshots=%d ...'%nshots)

task = classical_device.run(ahs_program, shots=nshots)

# The result can be downloaded directly into an object in the python session:
result = task.result()

print('\nM:dump localSimu results:",result.get_counts()')

#  Let's have a look at the average density of atoms on each site:


from quera_ahs_utils.analysis import get_avg_density

n_rydberg = get_avg_density(result)
print('\nM:dump probs:\n',n_rydberg.tolist())

# ### Simulation on Aquila

# And now for the truly exciting part! Let's bring Aquila into the game.

print('M: connecting to Aquila....')

from braket.aws import AwsDevice
aquila = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")

print(aquila)

print('M: submit task to Aquila...')
# To make the program compatible with the quantum hardware, we still need to slice it into discrete time steps:
discretized_ahs_program = ahs_program.discretize(aquila)

task = aquila.run(discretized_ahs_program, shots=nshots)
print('M:task submitted')
result = task.result()
n_rydberg = get_avg_density(result)
