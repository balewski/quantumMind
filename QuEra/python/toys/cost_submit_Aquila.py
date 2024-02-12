#!/usr/bin/env python3

# taken from https://docs.aws.amazon.com/braket/latest/developerguide/braket-pricing.html

#import any required modules
from braket.aws import AwsDevice

from braket.tracking import Tracker
from braket.devices import LocalSimulator
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.hamiltonian import Hamiltonian
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from quera_ahs_utils.drive import get_drive
import numpy as np
from pprint import pprint
from decimal import Decimal

def ahs_problem():
    a = 6.1e-6  # meters
    N_atoms = 11
    register = AtomArrangement()
    for i in range(N_atoms):   register.add([0.0, i*a])

    print('\nM:dump atom coordinates')
    xL=register.coordinate_list(0)
    yL=register.coordinate_list(1)
    
    for x,y in zip(xL,yL):  print(x,y)
    
    omega_max = 2e6 * 2 * np.pi
    detuning_min = -9e6 * 2 * np.pi
    detuning_max = 7e6 * 2 * np.pi
    
    time_max = Decimal('4')/1000000 # 4 us
    time_ramp =time_max/4

    # rounding is required for Aquila
    omega_max=int(omega_max/400)*400
    detuning_min=int(detuning_min*5)/5
    detuning_max=int(detuning_max*5)/5
    
    time_points = [0, time_ramp, time_max - time_ramp, time_max]
    omega_values = [0, omega_max, omega_max, 0]
    detuning_values = [detuning_min, detuning_min, detuning_max, detuning_max]
    phase_values = [0, 0, 0, 0]

    H = Hamiltonian()
    drive = get_drive(time_points, omega_values, detuning_values, phase_values)
    H += drive
    program = AnalogHamiltonianSimulation(  register=register, hamiltonian=H)
    
    print('\nM:dump drive Amplitude :\n',drive.amplitude.time_series.times(), drive.amplitude.time_series.values())


    return program
    

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    device = LocalSimulator("braket_ahs")
    #device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
    #device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")

    ahs_prog=ahs_problem()
    shots=30
    with Tracker() as tracker:
        print('executing  shots=%d...'%shots)
        task = device.run(ahs_prog, shots=shots)
        result=task.result()  # this forces wait for task completion/terminationcost tracking 
        if task.state()=='COMPLETED': 
            rawBitstr=result.get_counts()
            print('COMPLETED, rawBitstr:');  pprint(rawBitstr)
           

#Your results
print('status=%s, cost tracker, charge/$:'%task.state(),tracker.simulator_tasks_cost())
pprint(tracker.quantum_tasks_statistics())


