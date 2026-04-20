#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Demonstrator crashing on Braket emulator for 10 atoms system 
https://github.com/aws/amazon-braket-examples/issues 

Hi Jan, this bug is due to the integrator taking intermediate steps outside of the range of interpolated values that are computed before the integration starts. I'll take a look at what needs to be changed to correct this error. In the meantime, can you try using device.run(..., solver_method="bdf") 
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
def placeAtoms(nAtom):
        register = AtomArrangement()
        dist=Decimal('20.e-6')
              
        numY=4
        print('\nplace %d atoms\ni j  x(m)     y(m)  :'%nAtom)
        for k in range(nAtom):
            i=k//numY ; j=k%numY 
            x=i*dist ;  y=j*dist
            print(i,j,x,y)
            register.add([x,y])
        return register

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
    nAtom=16
    t_max=Decimal('4e-6') # evolution time
    shots=1000

    atoms=placeAtoms(nAtom)
    H=buildHamiltonian(t_max)
    ahs_program = AnalogHamiltonianSimulation(
            hamiltonian=H,  register=atoms )

    device = LocalSimulator("braket_ahs")
    evolSteps=int(t_max/Decimal('0.01e-6'))
    #evolSteps=200
    print('\nRun emulator: nAtom=%d, evol time %.3f us , steps=%d ...'%(nAtom,float(t_max)*1e6,evolSteps))
    t1=time.time()   
    job = device.run(ahs_program, shots=shots, steps=evolSteps, solver_method="bdf")
    t2=time.time()   
    
    rawCounts=job.result().get_counts()
    print('finised in %.1f sec, measured %d bits-strings, sample counts:'%(t2-t1,len(rawCounts)))
    for i,key in enumerate(rawCounts):
        print(key,rawCounts[key])
        if i >10: break
    print('Done')
