#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
https://github.com/aws/amazon-braket-examples/issues 

AtomArrangementValidator throws error for no reason.
None of atoms is closer than 4.8 um

The error is correct. I am not allowed to place atoms closer than 4 um along the y-direction.

1 validation error for AtomArrangementValidator
__root__
  Sites [Decimal('0.0000167'), Decimal('0.0000282')] and site [Decimal('0.0000064'), Decimal('0.0000316')] have y-separation (0.0000034). It must either be exactly zero or not smaller than 0.000004 meters (type=value_error)

'''

from pprint import pprint 
from decimal import Decimal
import numpy as np
import time

from braket.ahs.atom_arrangement import AtomArrangement
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField
from braket.ahs.hamiltonian import Hamiltonian
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.devices import LocalSimulator
 
#...!...!....................
def placeAtoms():
        posL=[[0.0, 0.0], [0.0, 5.7e-06], [0.0, 1.14e-05], [0.0, 1.71e-05], [0.0, 2.28e-05], [1.7613e-06, 2.82207e-05], [6.37317e-06, 3.157116e-05], [1.20726e-05, 3.157116e-05], [1.66839e-05, 2.82207e-05], [1.84452e-05, 2.28e-05], [1.84452e-05, 1.71e-05], [1.84452e-05, 1.14e-05], [1.84452e-05, 5.7e-06], [2.00919e-05, 3.16287e-05]]

        atoms = AtomArrangement()
        for x,y in posL:
                #print('xy',x,y)            
                atoms.add([Decimal(x),Decimal(y)])
        return atoms


#...!...!....................
def buildHamiltonian(t_max):        
        t_ramp = Decimal('1e-7') # 0.1 us

        omega_min = 0       
        omega_max = 1.8 * 2 * np.pi *1e6  # units rad/sec
        delta_max=0           # no detuning
        
        # constant Rabi frequency
       
        t_points = [0, t_ramp, t_max - t_ramp, t_max]
        omega_values = [omega_min, omega_max, omega_max, omega_min]
        Omega = TimeSeries()
        for t,v in zip(t_points,omega_values):
            bb=int(v/400)*Decimal('400')  
            Omega.put(t,bb)
            
        # all-zero phase and detuning Delta
        Phase = TimeSeries().put(0.0, 0.0).put(t_max, 0.0)  # (time [s], value [rad])
        Delta_global = TimeSeries().put(0.0, 0.0).put(t_max, delta_max)  # (time [s], value [rad/s])

        drive = DrivingField(
            amplitude=Omega,
            phase=Phase,
            detuning=Delta_global
        )
        H = Hamiltonian()
        H += drive
        print('\ndump drive filed Amplitude(time) :\n',drive.amplitude.time_series.times(), drive.amplitude.time_series.values())
        return H

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    # config
   
    t_max=Decimal('4e-6') # evolution time
    shots=10
    atoms=placeAtoms()
    H=buildHamiltonian(t_max)
    ahs_program = AnalogHamiltonianSimulation(
            hamiltonian=H,  register=atoms )

    from braket.aws import AwsDevice
    device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
    discr_ahs_program = ahs_program.discretize(device)
    
    evolSteps=200
    print('\nRun emulator:  evol time %.3f us , steps=%d ...'%(float(t_max)*1e6,evolSteps))
    t1=time.time()   
    job = device.run(discr_ahs_program, shots=shots, steps=evolSteps, solver_method="bdf")
    t2=time.time()   
    
    rawCounts=job.result().get_counts()
    print('finised in %.1f sec, measured %d bits-strings, sample counts:'%(t2-t1,len(rawCounts)))
    for i,key in enumerate(rawCounts):
        print(key,rawCounts[key])
        if i >10: break
    print('Done')
