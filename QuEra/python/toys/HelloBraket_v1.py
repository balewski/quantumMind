#!/usr/bin/env python3
# coding: utf-8


# ### Goal 
# 
# To say hello to the world of neutral atoms, let's investigate one of the central phenomena in quantum many-body physics: the emergence of **ordered phases of matter**. In this tutorial, we'll show how Aquila can prepare the simplest of such ordered phases, namely an antiferromagnetic phase (aka $\mathbb{Z_2}$ phase) in both one- and two-dimensional arrays of atoms. To make this possible, we will make use of the [Rydberg blockade](https://queracomputing.github.io/Bloqade.jl/dev/tutorials/1.blockade/main/) - the mechanism at the heart of our quantum computing architecture.

# ### Register
# 
# Let's start off with the 1D case.   To achieve our goal, we first need to define our Rydberg-atom quantum computer register. We arrange them in a line with a separation of 6.1μm between each pair of atoms. 


from braket.ahs.atom_arrangement import AtomArrangement
import numpy as np
from pprint import pprint as pp
from quera_ahs_utils.plotting import show_register
#import matplotlib.pyplot as plt

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
# The next component we need to specify is the Hamiltonian. It's the energy function that governs the behaviour of our atoms, including their interactions. 
# 
# In the lab, the Hamiltonian is implemented by applying lasers to the atoms. 



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


# ### Defining the program (1D case)
# 
# Now, we can combine the register and Hamiltonian into a program. In particular, this program falls within the class of Analog Hamiltonian Simulation (AHS). If you're curious about other types of quantum computing, take a look at [gate-based circuits](https://github.com/aws/amazon-braket-examples/blob/main/examples/getting_started/1_Running_quantum_circuits_on_simulators/1_Running_quantum_circuits_on_simulators.ipynb) or tutorials on [quantum annealing](https://github.com/aws/amazon-braket-examples/blob/main/examples/quantum_annealing/Dwave_TravelingSalesmanProblem/Dwave_TravelingSalesmanProblem.ipynb).



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
# Before submitting the task to run on actual quantum hardware, let's first check our program by running a local simulation on one of AWS's classical servers.


from braket.devices import LocalSimulator

classical_device = LocalSimulator("braket_ahs")

nshots = 6
task = classical_device.run(ahs_program, shots=nshots)

# The result can be downloaded directly into an object in the python session:
result = task.result()

print('\nM:dump localSimu results:",result.get_counts()')

# What phase of matter did we create? Let's have a look at the average density of atoms on each site:

# In[12]:


from quera_ahs_utils.analysis import get_avg_density
from quera_ahs_utils.plotting import plot_avg_density

n_rydberg = get_avg_density(result)
print('\nM:dump tolist:\n',n_rydberg.tolist())



# Indeed, we already see from this classical simulation that a pattern emerges: The alternating occupation density in the chain of atoms indicates a so-called $\mathbb{Z_2}$ phase. 

# ### Simulation on Aquila
# And now for the truly exciting part! Let's bring Aquila into the game.

from braket.aws import AwsDevice
if 0:
    from braket.aws import AwsSession
    from boto3 import Session

    boto_session = Session(region_name="us-east-1")
    aws_session = AwsSession(boto_session)
    print(aws_session)
    print('M:request access to aquila...')
    aquila = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila",aws_session)

aquila = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
    
print(aquila)

print('M: submit task to Aquila...')
# To make the program compatible with the quantum hardware, we still need to slice it into discrete time steps:
discretized_ahs_program = ahs_program.discretize(aquila)

task = aquila.run(discretized_ahs_program, shots=nshots)
result = task.result()
n_rydberg = get_avg_density(result)
